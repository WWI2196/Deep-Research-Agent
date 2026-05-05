# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development commands

```bash
# Python (uv)
uv sync                                    # install all deps + venv
uv sync --group dev                        # include dev deps (pytest, ruff, mypy)
uv run uvicorn src.backend.server:app --host 127.0.0.1 --port 8787  # start server

# Tests
uv run pytest tests/ -v                    # backend tests (156)
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
  - `server.py` — FastAPI app, 9 REST endpoints, SSE streaming
  - `config.py` — Configuration loading (env var > config.yaml > built-in default). `save_config()` preserves unmanaged sections (providers, search).
  - `models.py` — `ResearchState` TypedDict + Pydantic models
  - `graph.py` — LangGraph `StateGraph` with **7 async nodes**, `build_and_run_graph()`
  - `llm.py` — Unified async `chat()` routing with per-role provider selection + TTL caching
  - `agents.py` — Backward-compatible re-export shim (prefer importing from specific modules)
  - `helpers.py` — Pure text/JSON utilities (extract_json, clean_think_tags, needs_continuation, enforce_source_diversity, etc.)
  - `planning.py` — Planning-phase agents: `generate_research_plan` (structured JSON), `split_into_subtasks` (self-heal retry)
  - `subagent.py` — Subagent orchestration: rules-based `generate_search_queries`, `batch_evaluate_sources` (merged score+select), `run_subagent`, `run_subagents_parallel`
  - `synthesis.py` — Synthesis: `synthesize_report` (single-pass + truncation continuation), `add_citations` (rule-based [src: url] → [^n])
  - `search.py` — SearXNG search + trafilatura content extraction
  - `prompts.py` — System prompt templates for 6 roles + evaluation
  - `persistence.py` — SQLite persistence (runs, checkpoints, sources, subagent_reports). All I/O via `asyncio.to_thread`.
  - `export.py` — Markdown export
  - `providers/` — `OpenAICompatibleProvider` and `AnthropicProvider` (+ base class with exponential backoff)
- `src/renderer/` — Frontend UI (Vanilla JS, no framework), served as static files by FastAPI
  - Components: `input.js`, `dashboard.js`, `phases.js`, `subagents.js`, `sources.js`, `report.js`, `settings.js`, `history.js`
  - Infrastructure: `app.js` (router + page lifecycle), `api.js` (HTTP+SSE), `store.js` (`createStore()` event-driven state)
- `tests/` — pytest suite (156 backend + 49 frontend vitest)
  - `test_providers/` — Provider tests (anthropic, openai_compatible, factory)
  - `frontend/` — Vitest + jsdom frontend tests (store, format, markdown)

**Pipeline**: LangGraph `StateGraph` with **7 async nodes** in [src/backend/graph.py](src/backend/graph.py). Flow: `init → plan → split → subagents → reflection → (loop or proceed) → synthesize → cite → END`. Scale node removed; budget per subtask via `estimated_searches` field.

**State**: `ResearchState` (TypedDict in [src/backend/models.py](src/backend/models.py)) — holds query, plan (dict), subtasks, subagent reports, sources, iteration count, cited report, memory. `research_plan` is now a structured dict with `dimensions`, `output_structure`, `methodology`.

**LLM routing** ([src/backend/config.py](src/backend/config.py)): Priority chain: env var > `~/.deep-research/config.yaml` > built-in default. 7 roles (planner, splitter, subagent, evaluator, coordinator, reflection, citation) can each use a different provider+model. Config supports `${VAR}` env substitution. Six built-in providers: mimo, openai, anthropic, gemini, deepseek, openrouter.

**Search** ([src/backend/search.py](src/backend/search.py)): SearXNG (self-hosted, 70+ engines aggregated) via JSON API at `http://127.0.0.1:8080`. No API key needed. Content extraction via trafilatura (free, pip-installable) — fetches page HTML and extracts clean markdown.

**Agents** ([src/backend/subagent.py](src/backend/subagent.py), [src/backend/planning.py](src/backend/planning.py), [src/backend/synthesis.py](src/backend/synthesis.py)):
- `generate_research_plan`: Structured JSON output with dimensions (name, scope, keywords, source_types) + output_structure
- `split_into_subtasks`: Self-heal — JSON parse failure feeds error back to LLM for retry; falls back to dimension-per-subtask
- `generate_search_queries`: **Rules-based** (no LLM) — keywords × source_type modifiers. Fallback to title if no keywords.
- `batch_evaluate_sources`: Single LLM call per batch — scores quality AND decides full_text in one response (merged SOURCE_EVALUATE prompt)
- `run_subagent`: 6-step flow — rules queries → SearXNG search → evaluate+select → trafilatura extract → build evidence → write report with `[src: url]` markers. Empty report retried once.
- `synthesize_report`: Single-pass LLM with max_tokens=16384, 6-round truncation continuation, fallback to concatenation
- `add_citations`: **Rule-based** (no LLM) — parses `[src: url]`, deduplicates, assigns `[^n]`, generates References
- `_continue_if_truncated`: Detects missing end_marker, sends continuation prompts with tail context
- `_refine_queries_if_needed`: Re-searches when avg quality < 0.5 and < 3 high-quality sources
- `_enforce_source_diversity`: Caps 3 sources per domain

**Persistence** ([src/backend/persistence.py](src/backend/persistence.py)): Local SQLite at `~/.deep-research/history.db`. Four tables: `runs`, `checkpoints`, `sources`, `subagent_reports`. Checkpoints written after every pipeline phase. Sources and subagent reports persisted during subagent completion.

**SSE streaming** ([src/backend/server.py](src/backend/server.py)): `POST /api/research/stream` returns SSE stream. Events flow through `asyncio.Queue`. `_translate_event()` maps internal events to wire format.

**Parallelism**: Subagents run concurrently via `asyncio.gather`. Search and extract also use `asyncio.gather` (with `asyncio.to_thread` for sync trafilatura I/O). SQLite I/O via `asyncio.to_thread`.

**Frontend** ([src/renderer/](src/renderer/)): Single-page app with 5 pages. State via `createStore()` — `store.get/set/subscribe/reset`. Page lifecycle via `onPageCleanup(page, fn)` — navigates clean up listeners and timers. Markdown via CDN-loaded `marked.js`.

**Config file** ([config.yaml.example](config.yaml.example)): Template for `~/.deep-research/config.yaml`. `save_config()` preserves unmanaged sections (providers, search) when writing.

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
- All changes must maintain 156 Python + 49 frontend tests passing
- trafilatura is the only content extraction method — no paid alternatives
- `save_config()` must preserve `providers` and other unmanaged sections in config.yaml
