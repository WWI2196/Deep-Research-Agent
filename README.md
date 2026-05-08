# Deep Research Agent

A multi-agent deep research system that plans, searches, reads, reflects, and synthesizes high-quality cited reports.

**Web app**: Python FastAPI + LangGraph backend, Vanilla JS browser frontend. Self-hosted search, local document library RAG, zero paid scraping APIs.

## Quick Start

### Prerequisites

- Python 3.12+ with [uv](https://docs.astral.sh/uv/)
- Docker (for SearXNG)
- An LLM API key (OpenRouter, Mimo, or any OpenAI-compatible provider)

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
# Create docker-compose.yml and settings.yml (see docs/searxng-setup.md)
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
init → plan → split → subagents → reflection → synthesize → cite → END
```

- **plan**: Structured research plan (dimensions + keywords + source types)
- **split**: Break into 3–8 parallel subtasks
- **subagents**: Each subagent searches the web + your document library, evaluates sources, extracts content, and writes a cited report
- **reflection**: Quantitative per-dimension scoring; gaps trigger targeted re-search (up to max_iterations)
- **synthesize**: Single-pass LLM synthesis with automatic truncation recovery
- **cite**: Rule-based citation numbering + concurrent URL liveness verification

## Document Library (RAG)

Upload PDF, DOCX, TXT, MD, or HTML files into collections. The system uses:
- **Chroma** vector store (semantic search)
- **bm25s** + jieba (Chinese keyword search)
- **RRF fusion** to combine both

Documents are parsed, chunked, embedded, and indexed asynchronously. During research, your document library is searched in parallel with web search.

## Testing

```bash
uv run pytest tests/ -v    # 194 backend tests
```

## License

MIT
