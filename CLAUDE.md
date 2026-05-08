# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development commands

```bash
# Python (uv)
uv sync                                    # install all deps + venv
uv sync --group dev                        # include dev deps (pytest, ruff, mypy)
uv run uvicorn src.backend.server:app --host 127.0.0.1 --port 8787  # start server

# Tests
uv run pytest tests/ -v                    # backend tests (194)
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
- `src/backend/` — Python research engine (FastAPI + LangGraph + SQLite + Chroma), also serves static frontend files
  - `server.py` — FastAPI app, REST endpoints + SSE streaming. 新增 9 个文档库 API endpoints。
  - `config.py` — Configuration loading (env var > config.yaml > built-in default). `save_config()` preserves unmanaged sections (providers, search). New: `context_compress_retries`, `keep_tool_results`.
  - `models.py` — `ResearchState` TypedDict + Pydantic models. New: `document_collections` 字段。
  - `graph.py` — LangGraph `StateGraph` with **7 async nodes**, `build_and_run_graph()`
  - `llm.py` — Unified async `chat()` routing with per-role provider selection + TTL caching
  - `agents.py` — Backward-compatible re-export shim (prefer importing from specific modules)
  - `helpers.py` — Pure text/JSON utilities (extract_json, clean_think_tags, needs_continuation, enforce_source_diversity, query_similarity, generate_broader_queries, etc.). New: `normalize_search_item` 保留原始 `source` 字段。
  - `planning.py` — Planning-phase agents: `generate_research_plan` (structured JSON), `split_into_subtasks` (self-heal retry)
  - `subagent.py` — Subagent orchestration: fuzzy-dedup `generate_search_queries`, `batch_evaluate_sources` (merged score+select), `run_subagent` (query cache, empty-result rollback, evidence compression, [src:] marker validation), `run_subagents_parallel`. New: `_search_document_collections()` 并行文档库混合检索。
  - `synthesis.py` — Synthesis: `synthesize_report` (single-pass + truncation continuation + failure summary retry), `add_citations` (rule-based [src: url] → [^n] + URL liveness verification + 自动清理重复 References), `_generate_failure_summary`
  - `search.py` — SearXNG search + trafilatura content extraction
  - `prompts.py` — System prompt templates for 6 roles + evaluation. Subagent/Synthesis prompt 已移除自动 Sources/References 生成，改为纯内联引用。
  - `persistence.py` — SQLite persistence (runs, checkpoints, sources, subagent_reports, **collections, documents**). All I/O via `asyncio.to_thread`.
  - `export.py` — Markdown export
  - `document_store.py` — **NEW** — Chroma + bm25s + RRF 混合检索核心。`DocumentStore` 类封装向量存储、关键词索引、文档管理。支持异步解析、重新索引、category 过滤。
  - `document_parser.py` — **NEW** — PDF/DOCX/TXT/MD/HTML 统一解析入口
  - `tracing.py` — **NEW** — Structured tracing for research runs. `trace()` / `trace_llm_call()` use `contextvars` to propagate `run_id` implicitly through async call stacks. Log levels: debug/info/warning/error, controlled by `AppConfig.log_level`.
  - `providers/` — `OpenAICompatibleProvider` and `AnthropicProvider` (+ base class with exponential backoff)
- `src/renderer/` — Frontend UI (Vanilla JS, no framework), served as static files by FastAPI
  - Components: `input.js`, `dashboard.js`, `phases.js`, `subagents.js`, `sources.js`, `report.js`, `settings.js`, `history.js`
  - **New**: `library.js` — 文档库管理页面（创建/删除/上传/重索引）
  - **New**: `log-viewer.js` — Debug log viewer with phase/type filters, search, expandable details, and LLM-call latency display. Attachable to report page and history rows.
  - Infrastructure: `app.js` (router + page lifecycle), `api.js` (HTTP+SSE), `store.js` (`createStore()` event-driven state)
- `tests/` — pytest suite (194 backend + 49 frontend vitest)
  - `test_providers/` — Provider tests (anthropic, openai_compatible, factory)
  - `frontend/` — Vitest + jsdom frontend tests (store, format, markdown)
  - **New**: `test_document_store.py` — DocumentStore CRUD + 混合检索测试
  - **New**: `test_tracing.py` — Tracing contextvar and trace log level filtering tests

**Pipeline**: LangGraph `StateGraph` with **7 async nodes** in [src/backend/graph.py](src/backend/graph.py). Flow: `init → plan → split → subagents → reflection → (loop or proceed) → synthesize → cite → END`. Scale node removed; budget per subtask via `estimated_searches` field. New: synthesis retry on truncation/low-quality with failure summary; cite node runs concurrent URL liveness check.

**State**: `ResearchState` (TypedDict in [src/backend/models.py](src/backend/models.py)) — holds query, plan (dict), subtasks, subagent reports, sources, iteration count, cited report, memory, query_cache, synthesis_retry_count, context_compress_retries, keep_tool_results, **document_collections**. `research_plan` is now a structured dict with `dimensions`, `output_structure`, `methodology`.

**LLM routing** ([src/backend/config.py](src/backend/config.py)): Priority chain: env var > `~/.deep-research/config.yaml` > built-in default. 7 roles (planner, splitter, subagent, evaluator, coordinator, reflection, citation) can each use a different provider+model. Config supports `${VAR}` env substitution. Six built-in providers: mimo, openai, anthropic, gemini, deepseek, openrouter.

**Search** ([src/backend/search.py](src/backend/search.py)): SearXNG (self-hosted, 70+ engines aggregated) via JSON API at `http://127.0.0.1:8080`. No API key needed. Content extraction via trafilatura (free, pip-installable) — fetches page HTML and extracts clean markdown.

**RAG / Document Library** ([src/backend/document_store.py](src/backend/document_store.py)):
- **向量数据库**: Chroma (文件持久化，`~/.deep-research/chroma/`)
- **Embedding**: BAAI/bge-small-zh-v1.5 (384维, ~50MB, 中文优化)
- **关键词检索**: bm25s + jieba 中文分词
- **融合策略**: RRF (Reciprocal Rank Fusion, k=60)
- **混合检索流程**: Chroma 向量检索 + bm25s 关键词检索 → RRF 融合 → Top-K chunks
- **Pipeline 集成**: subagent 阶段并行执行 SearXNG 搜索和文档库混合检索。文档来源标记 `source: "document"`，quality_score 保底 0.85，跳过 trafilatura 直接以 chunk text 作为 evidence。
- **异步解析**: 上传 API 立即返回 pending，后台 asyncio task 执行解析→分块→嵌入→重建索引
- **重新索引**: 支持单文档重索引和全库重索引
- **Category 过滤**: 混合检索支持按 category metadata 过滤

**Agents** ([src/backend/subagent.py](src/backend/subagent.py), [src/backend/planning.py](src/backend/planning.py), [src/backend/synthesis.py](src/backend/synthesis.py)):
- `generate_research_plan`: Structured JSON output with dimensions (name, scope, keywords, source_types) + output_structure
- `split_into_subtasks`: Self-heal — JSON parse failure feeds error back to LLM for retry; falls back to dimension-per-subtask
- `generate_search_queries`: **Rules-based** (no LLM) — keywords × source_type modifiers + fuzzy Jaccard dedup. Fallback to title if no keywords.
- `batch_evaluate_sources`: Single LLM call per batch — scores quality AND decides full_text in one response (merged SOURCE_EVALUATE prompt)
- `run_subagent`: 6-step flow — rules queries → SearXNG search (query cache + empty-result rollback) + **文档库混合检索** → evaluate+select → trafilatura extract → build evidence (keep_tool_results compression) → write report with `[src: url]` markers. Retries on empty report or missing citations.
- `synthesize_report`: Single-pass LLM with max_tokens=16384, 6-round truncation continuation, failure_summary inject for retry. Fallback to concatenation.
- `add_citations`: **Rule-based** (no LLM) — parses `[src: url]`, normalizes URLs, deduplicates, assigns `[^n]`, generates References, concurrently verifies URL accessibility via trafilatura, marks unverified sources. **自动清理 LLM 生成的重复 References/Sources 小节**。
- `_generate_failure_summary`: Generates compact post-mortem for synthesis retry (what happened, covered, missing, remaining findings)
- `_verify_citation_urls`: Concurrent 8-way trafilatura fetch to check URL liveness (10s timeout each). `file://` 路径用 `Path.exists()` 验证。
- `_continue_if_truncated`: Detects missing end_marker, sends continuation prompts with tail context
- `_refine_queries_if_needed`: Re-searches when avg quality < 0.5 and < 3 high-quality sources
- `_enforce_source_diversity`: Caps 3 sources per domain.`file://` URL 以文件父路径作为 domain，避免同一库文档被误限。
- `query_similarity`: Jaccard word-token similarity for fuzzy query dedup
- `generate_broader_queries`: Strips modifiers for empty-result fallback queries

**Persistence** ([src/backend/persistence.py](src/backend/persistence.py)): Local SQLite at `~/.deep-research/history.db`. **Eight tables**: `runs`, `checkpoints`, `sources`, `subagent_reports`, `collections`, `documents`, `trace_logs`, `llm_calls`. Checkpoints written after every pipeline phase. Sources and subagent reports persisted during subagent completion. Trace logs and LLM call records written via `tracing.py` (contextvar-scoped).

**SSE streaming** ([src/backend/server.py](src/backend/server.py)): `POST /api/research/stream` returns SSE stream. Events flow through `asyncio.Queue`. `_translate_event()` maps internal events to wire format.

**Parallelism**: Subagents run concurrently via `asyncio.gather`. Search and extract also use `asyncio.gather` (with `asyncio.to_thread` for sync trafilatura I/O). SQLite I/O via `asyncio.to_thread`. Document parsing runs in background `asyncio.Task` with per-collection `asyncio.Lock`.

**Frontend** ([src/renderer/](src/renderer/)): Single-page app with **6 pages** (Home, Dashboard, Report, History, **Library**, Settings). State via `createStore()` — `store.get/set/subscribe/reset`. Page lifecycle via `onPageCleanup(page, fn)` — navigates clean up listeners and timers. Markdown via CDN-loaded `marked.js`.

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
- All changes must maintain 194 Python + 49 frontend tests passing
- trafilatura is the only content extraction method — no paid alternatives
- `save_config()` must preserve `providers` and other unmanaged sections in config.yaml
