# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development commands

```bash
# Python (uv)
uv sync                                    # install all deps + venv
uv sync --group dev                        # include dev deps (pytest, ruff, mypy)
uv run uvicorn src.backend.server:app --host 127.0.0.1 --port 8787  # start server

# Tests
uv run pytest tests/ -v                    # backend tests (160)
uv run pytest tests/test_agents.py -v      # single test file
npx vitest run                             # frontend tests (49)

# Lint & type check
uv run ruff check src/                     # lint
uv run mypy src/ --strict                  # type check (not yet passing)

# SearXNG (required for search)
cd ~/searxng && docker compose up -d       # start search engine
curl "http://127.0.0.1:8080/search?q=test&format=json"  # verify
```

## Architecture

**Product**: Web app (Python FastAPI backend + browser frontend). Users `git clone`, configure `~/.deep-research/config.yaml`, start SearXNG, run the server, open browser. Reference UI: Codex for Mac — dark, minimal, clean typography.

**Directory layout**:
- `src/backend/` — Python research engine (FastAPI + LangGraph + SQLite), also serves static frontend files
  - `server.py` — FastAPI app, 10 REST endpoints, SSE streaming
  - `config.py` — Configuration loading (env var > config.yaml > built-in default)
  - `models.py` — `ResearchState` TypedDict + Pydantic models
  - `graph.py` — LangGraph `StateGraph` with 8 async nodes, `build_and_run_graph()`
  - `llm.py` — Unified async `chat()` routing with per-role provider selection + caching
  - `agents.py` — Backward-compatible re-export shim (prefer importing from specific modules)
  - `helpers.py` — Pure text/JSON utilities (extract_json, clean_think_tags, needs_continuation, etc.)
  - `planning.py` — Planning-phase agents: `generate_research_plan`, `split_into_subtasks`, `compute_scaling`
  - `subagent.py` — Subagent orchestration: `run_subagent`, `run_subagents_parallel`, `batch_evaluate_sources`
  - `synthesis.py` — Synthesis & citation: `synthesize_report`, `add_citations`, `_continue_if_truncated`
  - `search.py` — SearXNG search + trafilatura content extraction
  - `prompts.py` — System prompt templates for all 9 roles
  - `persistence.py` — SQLite persistence (runs, checkpoints, sources, subagent_reports)
  - `export.py` — Markdown export
  - `providers/` — `OpenAICompatibleProvider` and `AnthropicProvider` (+ base class)
- `src/renderer/` — Frontend UI (Vanilla JS, no framework), served as static files by FastAPI
  - Components: `input.js`, `dashboard.js`, `phases.js`, `subagents.js`, `sources.js`, `report.js`, `settings.js`, `history.js`
  - Infrastructure: `app.js` (router), `api.js` (HTTP+SSE), `store.js` (event-driven state)
- `tests/` — pytest suite (160 backend + 49 frontend vitest)
  - `test_providers/` — Provider tests (anthropic, openai_compatible, factory)
  - `frontend/` — Vitest + jsdom frontend tests (store, format, markdown)

**Pipeline**: LangGraph `StateGraph` with 8 async nodes in [src/backend/graph.py](src/backend/graph.py). Flow: `init → plan → split → scale → subagents → reflection → (loop or proceed) → synthesize → cite → END`. Built and invoked per-request via `build_and_run_graph()`.

**State**: `ResearchState` (TypedDict in [src/backend/models.py](src/backend/models.py)) — holds query, plan, subtasks, subagent reports, sources, iteration count, cited report, memory.

**LLM routing** ([src/backend/config.py](src/backend/config.py)): Priority chain: env var > `~/.deep-research/config.yaml` > built-in default. 8 roles (planner, splitter, scaler, subagent, evaluator, coordinator, reflection, citation) can each use a different provider+model. Config supports `${VAR}` env substitution. Six built-in providers: mimo, openai, anthropic, gemini, deepseek, openrouter. Two provider types: `OpenAICompatibleProvider` (openai type) and `AnthropicProvider` (anthropic type), both fully async with exponential backoff retry ([src/backend/providers/base.py](src/backend/providers/base.py)).

**Search** ([src/backend/search.py](src/backend/search.py)): SearXNG (self-hosted, 70+ engines aggregated) via JSON API at `http://127.0.0.1:8080`. No API key needed. Content extraction via trafilatura (free, pip-installable) — fetches page HTML and extracts clean markdown. No paid search dependencies.

**Agents** ([src/backend/subagent.py](src/backend/subagent.py), [src/backend/planning.py](src/backend/planning.py), [src/backend/synthesis.py](src/backend/synthesis.py)): All LLM calls go through async `chat(role, messages)` in [src/backend/llm.py](src/backend/llm.py) with per-role provider routing and 5-min TTL cache. Key behaviors:
- `run_subagent`: 7-step flow — generate queries → SearXNG search → evaluate sources → LLM selects URLs for full-text → trafilatura extract → build evidence (FULL-TEXT / SNIPPET) → write report
- `_continue_if_truncated`: Detects cut-off output via dangling connectives, requests continuation
- `_refine_queries_if_needed`: Re-searches when avg quality < 0.5 and < 3 high-quality sources
- `_enforce_source_diversity`: Caps 3 sources per domain
- `batch_evaluate_sources`: Scores in batches of 20 via evaluator LLM role

**Persistence** ([src/backend/persistence.py](src/backend/persistence.py)): Local SQLite at `~/.deep-research/history.db`. Four tables: `runs`, `checkpoints`, `sources`, `subagent_reports`. Checkpoints written after every pipeline phase.

**SSE streaming** ([src/backend/server.py](src/backend/server.py)): `POST /api/research/stream` returns SSE stream. Events flow through `asyncio.Queue`, drained in main loop. `_translate_event()` maps internal events to wire format consumed by frontend.

**Parallelism**: Subagents run concurrently via `asyncio.gather`. Search and extract inside each subagent also use `asyncio.gather` (with `asyncio.to_thread` for sync trafilatura I/O).

**Frontend** ([src/renderer/](src/renderer/)): Single-page app with 5 pages — input, dashboard, report, history, settings. Markdown rendered via CDN-loaded `marked.js`. State management via `STATE` object + event-driven derivation. Served as static files by FastAPI, communicates with backend via same-origin fetch + SSE.

**Config file** ([config.yaml.example](config.yaml.example)): Template for `~/.deep-research/config.yaml`. Supports per-role model overrides, `${VAR}` env substitution, research default parameters. No search API keys needed — uses self-hosted SearXNG.

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
| `DELETE` | `/api/research/{id}` | Delete research run |
| `GET` | `/api/research/history` | Past research runs |
| `GET` | `/api/research/{id}/report` | Get report by run_id |

## Key constraints

- SearXNG must be running for search to work (`docker compose up -d` in `~/searxng/`)
- Start server with `uv run uvicorn src.backend.server:app --host 127.0.0.1 --port 8787`, then open `http://127.0.0.1:8787` in browser
- All Python changes must maintain 160 passing tests
- trafilatura is the only content extraction method — no paid alternatives
