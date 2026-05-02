# Deep Research Agent

A multi-agent deep research system that plans, searches, reads, reflects, and synthesizes high-quality cited reports. Desktop app powered by Electron + Python LangGraph backend.

## Features

- **8-node LangGraph pipeline**: `init → plan → split → scale → subagents → reflection → synthesize → cite`
- **Parallel subagents**: multiple research angles investigated concurrently via `asyncio.gather`
- **Reflection loop**: audits coverage gaps, spawns follow-up subtasks until `max_iterations` reached
- **SearXNG search**: self-hosted, 70+ engines aggregated (Google, Bing, Wikipedia, arXiv, etc.) — free, unlimited
- **trafilatura extraction**: free content fetching with clean markdown output — no paid scraping APIs
- **LLM URL selection**: subagent LLM decides which search results are worth full-text reading
- **Citation pass**: dedicated stage aligns claims to sources and generates inline references
- **Real-time SSE streaming**: all pipeline events pushed to frontend via `POST /api/research/stream`
- **Multi-provider LLM routing**: 8 roles can use different models; 6 built-in providers (mimo, openai, anthropic, gemini, deepseek, openrouter)
- **SQLite persistence**: runs, checkpoints, sources, reports stored locally at `~/.deep-research/history.db`
- **160 tests**: pytest + pytest-asyncio covering all backend modules

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    Electron Shell                    │
│  ┌──────────┐  ┌─────────────────────────────────┐  │
│  │ main/    │  │ renderer/                        │  │
│  │ python.ts │  │ Vanilla JS + marked.js           │  │
│  │ index.ts  │  │ SSE events → dashboard UI        │  │
│  └────┬─────┘  └──────────────┬──────────────────┘  │
│       │ spawns                │ fetch + EventSource  │
└───────┼───────────────────────┼──────────────────────┘
        │                       │
        ▼                       ▼
┌───────────────────────────────────────────────────────┐
│                Python Backend (:8787)                  │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌─────────┐  │
│  │ server.py│ │ graph.py │ │agents.py │ │search.py│  │
│  │ FastAPI  │ │LangGraph │ │ 8 roles  │ │SearXNG  │  │
│  │ SSE      │ │8 nodes   │ │_chat()   │ │trafilatura│ │
│  └──────────┘ └──────────┘ └──────────┘ └─────────┘  │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐              │
│  │config.py │ │persist.py│ │prompts.py│              │
│  │yaml+env  │ │ SQLite   │ │9 prompts │              │
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
- Node.js (for Electron frontend)
- An LLM API key (OpenRouter recommended, or any OpenAI-compatible provider)

### 1. Clone and install

```bash
git clone <repo-url> && cd deep-research-agent
uv sync
npm install
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
  max_iterations: 3
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
# Terminal 1: Backend
uv run uvicorn src.backend.server:app --host 127.0.0.1 --port 8787

# Terminal 2: Frontend (Electron)
npm run dev
```

## Pipeline

```
User Query
  │
  ▼
init ──── initialise state, assign run_id
  │
  ▼
plan ──── generate research strategy
  │
  ▼
split ─── break plan into 3–8 parallel subtasks
  │
  ▼
scale ─── estimate complexity, set search budget
  │
  ▼
subagents ──┬── subagent 1: search → select URLs → extract → report
            ├── subagent 2: search → select URLs → extract → report
            └── subagent N: search → select URLs → extract → report
  │
  ▼
reflection ── audit coverage, find gaps
  │           │
  │    gaps found ──→ create new subtasks ──→ subagents (loop)
  │
  ▼
synthesize ── merge all reports into one comprehensive article
  │
  ▼
cite ──────── add inline citations, generate references section
  │
  ▼
END ──────── stream final report via SSE
```

### Subagent detail

```
generate queries → SearXNG search (parallel)
  → batch evaluate source quality
  → enforce domain diversity
  → LLM selects URLs for full-text reading
  → trafilatura extract (parallel, markdown)
  → build evidence: [FULL-TEXT] + [SNIPPET]
  → write 800–1500 word report
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
uv run pytest tests/ -v    # 160 tests
```

| Module | Tests | Coverage |
|--------|-------|----------|
| config.py | 21 | ~90% |
| search.py | 8 | ~85% |
| agents.py | 60 | ~80% |
| graph.py | 10 | ~85% |
| server.py | 17 | ~85% |
| persistence.py | 9 | ~85% |
| models.py | 10 | ~85% |
| providers/ | 20 | ~90% |
| export.py | 3 | ~90% |
| integration | 3 | cross-module |

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python 3.12, FastAPI, LangGraph |
| Frontend | Vanilla JS, CSS, marked.js |
| Shell | Electron, TypeScript |
| Search | SearXNG (self-hosted Docker) |
| Extraction | trafilatura |
| LLM | OpenRouter / OpenAI-compatible / Anthropic |
| Storage | SQLite |
| Testing | pytest, pytest-asyncio |
| Package | uv (pyproject.toml) |

## License

MIT
