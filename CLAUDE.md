# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development commands

```bash
# Python (uv)
uv sync                                    # install all deps + venv
uv sync --group dev                        # include dev deps (ruff, mypy)
uv run uvicorn src.backend.server:app --host 127.0.0.1 --port 8787  # start server

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
- `src/backend/` — Python research engine (FastAPI + LangGraph + SQLite + Chroma), also serves static frontend files
  - `server.py` — FastAPI app, REST endpoints + SSE streaming. 23 endpoints total (9 research + 12 document + 2 tracing).
  - `config.py` — Configuration loading (env var > config.yaml > built-in default). `save_config()` preserves unmanaged sections (providers, search). New: `context_compress_retries`, `keep_tool_results`.
  - `models.py` — `ResearchState` TypedDict + Pydantic models. New: `document_collections` field.
  - `graph.py` — LangGraph `StateGraph` with **7 async nodes**, `build_and_run_graph()`. State maintains `_subtask_report_map: dict[str, str]` for precise deduplication when reflection re-runs subtasks.
  - `llm.py` — Unified async `chat()` routing with per-role provider selection + TTL caching
  - `agents.py` — Backward-compatible re-export shim (prefer importing from specific modules)
  - `helpers.py` — Pure text/JSON utilities (extract_json, clean_think_tags, needs_continuation, enforce_source_diversity, query_similarity, generate_broader_queries, etc.). New: `normalize_search_item` preserves original `source` field.
  - `planning.py` — Planning-phase agents: `generate_research_plan` (structured JSON), `split_into_subtasks` (self-heal retry)
  - `subagent.py` — Subagent orchestration: fuzzy-dedup `generate_search_queries`, `batch_evaluate_sources` (merged score+select), `run_subagent` (query cache, empty-result rollback, evidence compression, [src:] marker validation), `run_subagents_parallel`. New: `_search_document_collections()` parallel document hybrid retrieval.
  - `synthesis.py` — Synthesis: `synthesize_report` (single-pass + truncation continuation + failure summary retry), `add_citations` (rule-based [src: url] → [^n] + URL liveness verification + auto strip duplicate References), `_generate_failure_summary`, `_deepen_thin_sections` (expands thin sections with high-importance evidence). Section replacement ensures trailing `\n\n` to prevent heading粘连.
  - `search.py` — SearXNG search + dual-path content extraction (`requests` + `trafilatura` fast path; `Crawl4AI` browser-render fallback for JS-heavy pages)
  - `prompts.py` — System prompt templates for 6 roles + evaluation. Subagent/Synthesis prompts removed automatic Sources/References generation, use inline citations only.
  - `persistence.py` — SQLite persistence (runs, checkpoints, sources, subagent_reports, **collections, documents**). All I/O via `asyncio.to_thread`. `update_run_status` only sets `completed_at` for terminal states (completed/cancelled/failed).
  - `export.py` — Markdown export
  - `document_store.py` — **v3** Chroma + bm25s + RRF hybrid retrieval core. `DocumentStore` class encapsulates vector store, keyword index, document management. Supports async parsing, re-indexing, category filtering.
  - `document_parser.py` — **v3** PDF/DOCX/TXT/MD/HTML unified parsing entry
  - `tracing.py` — **v3** Structured tracing for research runs. `trace()` / `trace_llm_call()` use `contextvars` to propagate `run_id` implicitly through async call stacks. Log levels: debug/info/warning/error, controlled by `AppConfig.log_level`. `trace_llm_call` stores `error` as `None` when absent (not empty string) to avoid false positives in `error IS NOT NULL` queries.
  - `providers/` — `OpenAICompatibleProvider` and `AnthropicProvider` (+ base class with exponential backoff)
- `src/renderer/` — Frontend UI (Vanilla JS, no framework), served as static files by FastAPI
  - Components: `input.js`, `dashboard.js`, `phases.js`, `subagents.js`, `sources.js`, `report.js`, `settings.js`, `history.js`
  - **New**: `library.js` — Document library management page (create/delete/upload/reindex)
  - **New**: `log-viewer.js` — Debug log viewer with phase/type filters, search, expandable details, and LLM-call latency display. Attachable to report page and history rows.
  - Infrastructure: `app.js` (router + page lifecycle), `api.js` (HTTP+SSE), `store.js` (`createStore()` event-driven state)

**Pipeline**: LangGraph `StateGraph` with **7 async nodes** in [src/backend/graph.py](src/backend/graph.py). Flow: `init → plan → split → subagents → reflection → (loop or proceed) → synthesize → cite → END`. Scale node removed; budget per subtask via `estimated_searches` field. Reflection re-runs use `_subtask_report_map` for precise subtask_id-based deduplication. New: synthesis retry on truncation/low-quality with failure summary; cite node runs concurrent URL liveness check; `_deepen_thin_sections` expands thin sections with trailing newline guard.

**State**: `ResearchState` (TypedDict in [src/backend/models.py](src/backend/models.py)) — holds query, plan (dict), subtasks, subagent reports, sources, iteration count, cited report, memory, query_cache, synthesis_retry_count, context_compress_retries, keep_tool_results, **document_collections**. `research_plan` is now a structured dict with `dimensions`, `output_structure`, `methodology`. Additionally maintains `_subtask_report_map: dict[str, str]` (subtask_id → report) for precise deduplication when reflection re-runs subtasks.

**LLM routing** ([src/backend/config.py](src/backend/config.py)): Priority chain: env var > `~/.deep-research/config.yaml` > built-in default. 7 roles (planner, splitter, subagent, evaluator, coordinator, reflection, citation) can each use a different provider+model. Config supports `${VAR}` env substitution. Six built-in providers: mimo, openai, anthropic, gemini, deepseek, openrouter.

**Search** ([src/backend/search.py](src/backend/search.py)): SearXNG (self-hosted, 70+ engines aggregated) via JSON API at `http://127.0.0.1:8080`. No API key needed. Content extraction: `requests.get` + `trafilatura.extract` for fast path; `Crawl4AI` (`AsyncWebCrawler` with playwright Chromium / system Chrome fallback) for JS-rendered pages.

**RAG / Document Library** ([src/backend/document_store.py](src/backend/document_store.py)):
- **Vector DB**: Chroma (file-persistent, `~/.deep-research/chroma/`)
- **Embedding**: BAAI/bge-small-zh-v1.5 (384-dim, ~50MB, Chinese-optimized)
- **Keyword search**: bm25s + jieba Chinese tokenization
- **Fusion**: RRF (Reciprocal Rank Fusion, k=60)
- **Hybrid flow**: Chroma vector search + bm25s keyword search → RRF fusion → Top-K chunks
- **Pipeline integration**: subagent stage runs SearXNG search and document library hybrid retrieval in parallel. Document sources marked `source: "document"`, quality_score floor 0.85, skip trafilatura and use chunk text directly as evidence.
- **Async parsing**: Upload API returns pending immediately; background asyncio task performs parse → chunk → embed → rebuild index
- **Re-indexing**: Supports single-document and full-collection re-indexing
- **Category filter**: Hybrid search supports filtering by category metadata

**Agents** ([src/backend/subagent.py](src/backend/subagent.py), [src/backend/planning.py](src/backend/planning.py), [src/backend/synthesis.py](src/backend/synthesis.py)):
- `generate_research_plan`: Structured JSON output with dimensions (name, scope, keywords, source_types) + output_structure
- `split_into_subtasks`: Self-heal — JSON parse failure feeds error back to LLM for retry; falls back to dimension-per-subtask
- `generate_search_queries`: **Rules-based** (no LLM) — keywords × source_type modifiers + fuzzy Jaccard dedup. Fallback to title if no keywords.
- `batch_evaluate_sources`: Single LLM call per batch — scores quality AND decides full_text in one response (merged SOURCE_EVALUATE prompt)
- `run_subagent`: 6-step flow — rules queries → SearXNG search (query cache + empty-result rollback) + **document library hybrid search** → evaluate+select → `extract_async()` (trafilatura fast path + Crawl4AI browser fallback) → build evidence (keep_tool_results compression) → write report with `[src: url]` markers. Retries on empty report or missing citations.
- `synthesize_report`: Single-pass LLM with max_tokens=16384, 6-round truncation continuation, failure_summary inject for retry. Fallback to concatenation. `_deepen_thin_sections` expands thin sections with high-importance evidence; replacement ensures trailing `\n\n` to prevent heading粘连.
- `add_citations`: **Rule-based** (no LLM) — parses `[src: url]`, normalizes URLs, deduplicates, assigns `[^n]`, generates References, concurrently verifies URL accessibility via trafilatura, marks unverified sources. **Auto strips duplicate References/Sources sections generated by LLM**.
- `_generate_failure_summary`: Generates compact post-mortem for synthesis retry (what happened, covered, missing, remaining findings)
- `_verify_citation_urls`: Concurrent 8-way trafilatura fetch to check URL liveness (10s timeout each). `file://` paths checked with `Path.exists()`.
- `_continue_if_truncated`: Detects missing end_marker, sends continuation prompts with tail context
- `_refine_queries_if_needed`: Re-searches when avg quality < 0.5 and < 3 high-quality sources
- `_enforce_source_diversity`: Caps 3 sources per domain. `file://` URLs use file parent path as domain to avoid same-library documents being incorrectly limited.
- `query_similarity`: Jaccard word-token similarity for fuzzy query dedup
- `generate_broader_queries`: Strips modifiers for empty-result fallback queries

**Persistence** ([src/backend/persistence.py](src/backend/persistence.py)): Local SQLite at `~/.deep-research/history.db`. **Eight tables**: `runs`, `checkpoints`, `sources`, `subagent_reports`, `collections`, `documents`, `trace_logs`, `llm_calls`. Checkpoints written after every pipeline phase. Sources and subagent reports persisted during subagent completion. Trace logs and LLM call records written via `tracing.py` (contextvar-scoped).

**SSE streaming** ([src/backend/server.py](src/backend/server.py)): `POST /api/research/stream` returns SSE stream. Events flow through `asyncio.Queue`. `_translate_event()` maps internal events to wire format.

**Parallelism**: Subagents run concurrently via `asyncio.gather`. Search and extract also use `asyncio.gather` (with `asyncio.to_thread` for sync trafilatura I/O). SQLite I/O via `asyncio.to_thread`. Document parsing runs in background `asyncio.Task` with per-collection `asyncio.Lock`.

**Frontend** ([src/renderer/](src/renderer/)): Single-page app with **6 pages** (Home, Dashboard, Report, History, **Library**, Settings). State via `createStore()` — `store.get/set/subscribe/reset`. Page lifecycle via `onPageCleanup(page, fn)` — navigates clean up listeners and timers. Markdown via CDN-loaded `marked.js`.

**Dashboard event flow**: `input.js` starts SSE stream via `startResearchStream()`, then `navigateTo('dashboard')`. `dashboard.js` uses dual-track updates: `pollStatus()` (2s interval) for run status/progress + `pollTimeline()` (2s interval) for detailed phase/subagent events. SSE `complete` event is handled by `handleResearchEvent()` which sets `complete=true` and calls `stopPollers()`. The stream's `onDone` callback must invoke `handleResearchDone()` to check `complete` state and `navigateTo('report')` — this is the only path that guarantees navigation when SSE ends.

**Config file** ([config.yaml.example](config.yaml.example)): Template for `~/.deep-research/config.yaml`. `save_config()` preserves unmanaged sections (providers, search) when writing. `log_level` (info/debug/warning/error) controls tracing verbosity; default is "info".

**Tracing / Observability** ([src/backend/tracing.py](src/backend/tracing.py)):
- `trace()` writes structured trace logs (phase, event_type, message, details) for the current `run_id` propagated via `contextvars`
- `trace_llm_call()` records every LLM invocation (role, provider, model, messages, response, latency)
- Log level filtering: debug stores full messages; info stores truncated previews (120/200 chars); warning/error always persisted
- Frontend `log-viewer.js` renders timeline (trace + llm_calls merged), supports phase/type filtering, keyword search, and expandable JSON details

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
| `GET` | `/api/research/{id}/logs` | Get trace logs for a run (phase/level/type filters) |
| `GET` | `/api/research/{id}/timeline` | Merged timeline of trace logs + LLM calls |
| `GET` | `/api/collections` | List document collections |
| `POST` | `/api/collections` | Create collection |
| `DELETE` | `/api/collections/{id}` | Delete collection |
| `PATCH` | `/api/collections/{id}` | Update collection name/desc |
| `GET` | `/api/collections/{id}/documents` | List documents in collection |
| `POST` | `/api/collections/{id}/documents` | Upload document (async indexing) |
| `DELETE` | `/api/collections/{id}/documents/{doc_id}` | Delete document |
| `GET` | `/api/collections/{id}/documents/{doc_id}/download` | Download raw file |
| `POST` | `/api/collections/{id}/search` | Hybrid search within collection |
| `POST` | `/api/collections/{id}/reindex` | Re-index all documents |
| `POST` | `/api/collections/{id}/documents/{doc_id}/reindex` | Re-index single document |

## Key constraints

- SearXNG must be running for search to work (`docker compose up -d` in `~/searxng/`)
- Start server with `uv run uvicorn src.backend.server:app --host 127.0.0.1 --port 8787`, then open `http://127.0.0.1:8787` in browser
- Content extraction: `requests` + `trafilatura` fast path, `Crawl4AI` browser-render fallback for JS-heavy pages; no paid scraping APIs
- `save_config()` must preserve `providers` and other unmanaged sections in config.yaml
