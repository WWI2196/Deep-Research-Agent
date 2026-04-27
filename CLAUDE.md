# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development commands

```bash
# Backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run.py                          # starts FastAPI at http://localhost:8000

# Frontend
cd frontend && npm install
npm run dev                            # starts Next.js at http://localhost:3000
npm run lint                           # ESLint
npm run build                          # production build
```

No backend test suite exists. No type checker is configured for Python. The frontend has `npm run lint` (Next.js ESLint).

## Architecture

**Pipeline**: A LangGraph `StateGraph` with 8 nodes executes a research query end-to-end. The graph is built in [backend/graph.py](backend/graph.py) per request. Flow: `init → plan → split → scale → subagents → reflection → (loop or proceed) → synthesize → cite → END`. Reflection can loop back to subagents when gaps are found, up to `MAX_ITERATIONS`.

**State**: The graph operates on `ResearchState` (TypedDict in [backend/models.py](backend/models.py)) which holds the query, plan, subtasks, subagent reports, sources, iteration count, and final cited report.

**LLM routing** ([backend/config.py](backend/config.py)): `AppConfig` loads `LLM_PROVIDER`/`LLM_MODEL` from env with optional per-role overrides (`PLANNER_PROVIDER`, `SUBAGENT_MODEL`, etc.). Each of 8 roles can use a different provider+model. The abstract `LLMProvider` base ([backend/providers/base.py](backend/providers/base.py)) expects OpenAI-style message dicts and returns plain text. Four implementations: Gemini, OpenAI, Anthropic, and HuggingFace. The HuggingFace provider proxies multiple HF inference providers (novita, sambanova, auto) via `InferenceClient(provider=...)`.

**Two config files warning**: The root [config.py](config.py) defines `ModelConfig` classes with fallback chains per role using "novita"/"sambanova" providers — this is a standalone reference artifact, NOT imported by the backend. Runtime config lives in [backend/config.py](backend/config.py). The two are independent and may drift.

**SSE streaming** ([backend/server.py](backend/server.py), [backend/events.py](backend/events.py)): A thread-safe event bus (`emit`/`add_listener`) decouples pipeline nodes (which run in executor threads) from the SSE generator. `_translate_event()` maps internal event types to SSE wire events consumed by the frontend.

**Parallelism model**: Subagents run concurrently via `asyncio.to_thread()` in [backend/graph.py:30](backend/graph.py). Each subagent internally uses `ThreadPoolExecutor` for parallel search queries and URL extraction. This creates two levels of concurrency.

**Search layer** ([backend/search.py](backend/search.py)): Firecrawl primary → DuckDuckGo fallback (free, no API key). When Firecrawl fails with payment/auth errors, it's disabled for the session. `_normalise_search_response` handles both Pydantic v2 model responses and plain dicts.

**Agents** ([backend/agents.py](backend/agents.py)): All LLM calls go through `_chat(role, messages)` which emits telemetry events and retries on transient failures (exponential backoff, max 3 attempts). Key behaviors:
- `_continue_if_truncated`: Detects genuinely cut-off output by checking for dangling connectives ("and", "the", "of"...) at the end, then requests continuation up to 4 rounds. Avoids false positives from clean sentence endings.
- `_refine_queries_if_needed`: When average source quality score < 0.5 and fewer than 3 high-quality sources, generates refined search queries.
- `_enforce_source_diversity`: Caps 3 sources per domain for breadth.
- `batch_evaluate_sources`: Scores sources in batches of 20 via the evaluator LLM role.

**Persistence** ([backend/persistence.py](backend/persistence.py)): Best-effort Supabase writes for state checkpoints and artifacts (subagent reports, final report, cited report). Pipeline continues normally if Supabase is unreachable or unconfigured.

**Frontend**: Next.js 15 + React 19 + Tailwind CSS 4. Single-page app at [frontend/app/page.tsx](frontend/app/page.tsx) with three components: `ResearchChat` (SSE client + model selector), `SearchDisplay` (three-column dashboard with sticky sidebars), `MarkdownRenderer` (react-markdown + remark-gfm). The SSE client connects directly to `NEXT_PUBLIC_BACKEND_URL` (bypasses Next.js rewrite proxy because it buffers streaming responses in dev mode). All non-streaming calls go through Next.js proxy at `/api/*`.

**Supabase schema** ([supabase_schema.sql](supabase_schema.sql)): Three tables — `deep_research_runs` (upserted current state), `deep_research_checkpoints` (append-only phase history), `deep_research_artifacts` (reports and cited output).
