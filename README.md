# Deep Research Agent

A multi-agent deep research system that plans, searches, reads, reflects, and synthesizes high-quality cited reports. Web app powered by Python LangGraph backend with browser frontend.

## Features

- **7-node LangGraph pipeline**: `init → plan → split → subagents → reflection → synthesize → cite`
- **Structured planning**: Planner outputs JSON with dimensions, keywords, and source_types per dimension
- **Rules-based query generation**: Search queries derived from dimension keywords × source type modifiers (no LLM call)
- **Merged evaluation + selection**: Source scoring and full-text selection in a single LLM call
- **Parallel subagents**: multiple research angles investigated concurrently via `asyncio.gather`
- **Quantitative reflection**: Per-dimension 4-axis scoring (coverage/depth/evidence/recency), only low-score dimensions trigger gap-fill subtasks
- **Truncation recovery**: Multi-round continuation synthesis with auto-detection of cut-off output
- **Rule-based citations**: Subagent reports use `[src: url]` markers; deterministic `[^n]` numbering + References generation — zero LLM hallucination risk
- **SearXNG search**: self-hosted, 70+ engines aggregated (Google, Bing, Wikipedia, arXiv, etc.) — free, unlimited
- **trafilatura extraction**: free content fetching with clean markdown output — no paid scraping APIs
- **Real-time SSE streaming**: all pipeline events pushed to frontend via `POST /api/research/stream`
- **Multi-provider LLM routing**: 7 roles can use different models; 6 built-in providers (mimo, openai, anthropic, gemini, deepseek, openrouter)
- **SQLite persistence**: runs, checkpoints, sources, reports stored locally at `~/.deep-research/history.db`
- **205 tests**: 156 pytest + 49 vitest/jsdom frontend tests

## Architecture

```
Browser (http://127.0.0.1:8787)
  │
  │ fetch + SSE (same-origin)
  ▼
┌───────────────────────────────────────────────────────┐
│                Python Backend (:8787)                  │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌─────────┐  │
│  │ server.py│ │ graph.py │ │subagent  │ │search.py│  │
│  │ FastAPI  │ │LangGraph │ │planning  │ │SearXNG  │  │
│  │ Static   │ │7 nodes   │ │synthesis │ │trafilatura│ │
│  │ Files    │ │          │ │llm.py    │ │         │  │
│  └──────────┘ └──────────┘ └──────────┘ └─────────┘  │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐              │
│  │config.py │ │persist.py│ │prompts.py│              │
│  │yaml+env  │ │ SQLite   │ │6 prompts │              │
│  └──────────┘ └──────────┘ └──────────┘              │
└───────────────────────────────────────────────────────┘
        │
        ▼
┌──────────┐  ┌──────────┐
│ SearXNG  │  │ OpenRouter│
│ :8080    │  │ (or any   │
│ 70+ eng. │  │ OpenAI/   │
│ free     │  │ Anthropic │
└──────────┘  │ provider) │
              └──────────┘
```

## Quick Start

### Prerequisites

- Python 3.12+ with [uv](https://docs.astral.sh/uv/)
- Docker (for SearXNG)
- An LLM API key (OpenRouter recommended, or any OpenAI-compatible provider)
- A modern browser (Chrome, Firefox, Safari)

### 1. Clone and install

```bash
git clone <repo-url> && cd deep-research-agent
uv sync
```

### 2. Configure LLM

Create `~/.deep-research/config.yaml`:

```yaml
providers:
  openrouter:
    api_key: sk-or-v1-...

default:
  provider: openrouter
  model: anthropic/claude-sonnet-4

roles:
  planner:     { provider: openrouter, model: anthropic/claude-sonnet-4 }
  subagent:    { provider: openrouter, model: anthropic/claude-sonnet-4 }
  coordinator: { provider: openrouter, model: anthropic/claude-sonnet-4 }
  reflection:  { provider: openrouter, model: anthropic/claude-sonnet-4 }

research:
  max_iterations: 2
  quality_threshold: 0.7
```

Or use environment variables:

```bash
export OPENROUTER_API_KEY=sk-or-v1-...
export LLM_PROVIDER=openrouter
export LLM_MODEL=anthropic/claude-sonnet-4
```

### 3. Start SearXNG

```bash
mkdir -p ~/searxng
cd ~/searxng
# Create docker-compose.yml and settings.yml (see config.yaml.example)
docker compose up -d
# Verify: curl "http://127.0.0.1:8080/search?q=test&format=json"
```

### 4. Run

```bash
uv run uvicorn src.backend.server:app --host 127.0.0.1 --port 8787
# Then open http://127.0.0.1:8787 in your browser
```

## Pipeline

```
User Query
  │
  ▼
init ──────── initialise state, assign run_id
  │
  ▼
plan ──────── generate structured research plan (JSON: dimensions + keywords)
  │
  ▼
split ─────── break plan into 3–8 parallel subtasks (self-heal on JSON errors)
  │
  ▼
subagents ──┬── subagent 1: rules-based queries → search → evaluate+select → extract → report [src: url]
            ├── subagent 2: rules-based queries → search → evaluate+select → extract → report [src: url]
            └── subagent N: ...
  │
  ▼
reflection ── per-dimension 4-axis scoring (coverage/depth/evidence/recency)
  │           │
  │    gaps (<0.6) ──→ create targeted subtasks (max 3) ──→ subagents (loop)
  │
  ▼
synthesize ── single-pass LLM synthesis (max_tokens=16384) + truncation continuation
  │
  ▼
cite ──────── rule-based: parse [src: url] → assign [^n] → generate References
  │
  ▼
END ──────── stream final report via SSE
```

### Subagent detail

```
rules-based query generation (keywords × source_type modifiers)
  → SearXNG search (parallel, asyncio.gather)
  → batch evaluate + select full-text (single LLM call)
  → enforce domain diversity (max 3 per domain)
  → adaptive query refinement (if avg quality < 0.5)
  → trafilatura extract (parallel, markdown)
  → build evidence: [FULL-TEXT] + [SNIPPET]
  → write 800–1500 word report with [src: url] markers
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/health` | Backend health + provider info |
| `GET` | `/api/config` | Current configuration |
| `POST` | `/api/config` | Update configuration |
| `GET` | `/api/models` | Available LLM providers |
| `POST` | `/api/research` | Start research (returns run_id) |
| `POST` | `/api/research/stream` | Start research with SSE streaming |
| `POST` | `/api/research/{id}/cancel` | Cancel running research |
| `GET` | `/api/research/history` | Past research runs |
| `GET` | `/api/research/{id}/report` | Get report by run_id |

## Testing

```bash
uv run pytest tests/ -v    # 156 backend tests
npx vitest run             # 49 frontend tests
```

| Module | Tests |
|--------|-------|
| config.py | 21 |
| search.py | 7 |
| agent functions | 55 |
| graph.py | 10 |
| server.py | 17 |
| persistence.py | 9 |
| models.py | 10 |
| providers/ | 20 |
| export.py | 3 |
| integration | 3 |
| frontend (vitest) | 49 |
| **Total** | **205** |

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3.12, FastAPI, LangGraph |
| Frontend | Vanilla JS, CSS, marked.js |
| Search | SearXNG (self-hosted Docker) |
| Extraction | trafilatura |
| LLM | OpenRouter / OpenAI-compatible / Anthropic |
| Storage | SQLite |
| Testing | pytest, pytest-asyncio, vitest + jsdom |
| Package | uv (pyproject.toml) |

## License

MIT
