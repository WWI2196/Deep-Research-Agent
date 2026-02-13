# Deep Research Agent

<p align="center">
  <img src="assets/home.png" alt="Deep Research Agent" width="100%" />
</p>

A multi-agent, citation-aware research system that plans, delegates, reflects, and synthesizes high-quality reports in real time.

This project combines a FastAPI + LangGraph backend with a modern Next.js interface to deliver an end-to-end research workflow: from user query to structured, source-backed final report.

## Why this project

- Multi-agent orchestration: parallel subagents investigate different angles of the same problem.
- Reflection loop: the system audits its own coverage, detects gaps, and launches additional follow-up tasks when needed.
- Source-first workflow: Firecrawl-powered web search and extraction with source quality scoring, deduplication, and citation pass.
- Real-time observability: live progress phases, subagent lanes, LLM call telemetry, and source tracking in the UI.
- Multi-provider model routing: Gemini, OpenAI, Anthropic, and HuggingFace support with per-role overrides.

The result is a research assistant that is significantly more rigorous than a single-shot chat response and much better suited for deep, multi-step analysis.

---

## Core capabilities

### Deep Research algorithm 

```mermaid
flowchart TD
    U[User submits query to POST /api/research/stream] --> S[research_stream_generator starts SSE]
    S --> G[build_graph and compile LangGraph StateGraph]
    G --> I[init node]
    I --> P[plan node]
    P --> SP[split node]
    SP --> SC[scale node]
    SC --> SR[subagents node]

    SR --> F1[Select pending subtasks: id not in completed_subtasks]
    F1 --> F2[Emit subagents-launch with agent_details]
    F2 --> F3[Run one run_subagent per pending subtask via asyncio.to_thread]

    F3 --> A1[Per subagent: generate queries]
    A1 --> A2[Parallel Firecrawl search per query]
    A2 --> A3[Normalize and score sources via evaluator LLM]
    A3 --> A4{avg score < 0.5 and high quality < 3?}
    A4 -- yes --> A5[Generate refined queries and re-search]
    A5 --> A3
    A4 -- no --> A6[Enforce max 3 sources per domain and dedupe URLs]
    A6 --> A7[Parallel Firecrawl extract top URLs]
    A7 --> A8[Build evidence, fallback to snippets if extract empty]
    A8 --> A9[Write subagent report]
    A9 --> A10[Emit subagent-complete and return report/sources]

    A10 --> M1[Merge reports and deduped sources into state]
    M1 --> R[reflection node]

    R --> R0{iteration_count >= max_iterations?}
    R0 -- yes --> C1[Set research_complete true]
    R0 -- no --> R1[Reflection LLM audits reports and returns JSON subtasks]
    R1 --> R2{new subtasks returned?}
    R2 -- yes --> R3[Append gap subtasks and set research_complete false]
    R3 --> SR
    R2 -- no --> C1

    C1 --> SY[synthesize node]
    SY --> CI[cite node]
    CI --> O1[Emit final-result cited_report or report]
    O1 --> O2[Emit complete event with totals]

    subgraph Progress and UI updates
      E1[Backend emits phase and subagent events]
      E2[server translates event bus to SSE payloads]
      E3[Frontend renders phase timeline, agent lanes, live activity, source panels]
      E1 --> E2 --> E3
    end
```

### Deep Research in 10 quick steps

1. User sends a query to `POST /api/research/stream`.
2. Backend starts SSE and compiles the LangGraph state machine.
3. `plan` writes a research plan.
4. `split` creates independent subtasks.
5. `scale` sets subagent/tool/source budget.
6. `subagents` runs one agent per pending subtask in parallel.
7. Each subagent searches, scores sources, refines queries (if weak), extracts evidence, writes a report.
8. `reflection` audits coverage and either adds gap subtasks or marks research complete.
9. If gaps exist, only new/pending subtasks run in the next wave.
10. `synthesize` merges reports, `cite` adds citations, SSE returns final result + completion stats.

### Why this implementation is different

- Iterative, not one-shot: reflection can add new subtasks to close missing coverage.
- Deep parallelism: parallel subagents plus parallel search/extract inside each subagent.
- Evidence quality controls: source scoring, domain diversity cap, and query refinement loop.
- Transparent runtime: phase/subagent events are streamed live to the UI over SSE.
- Flexible model routing: different roles can use different providers/models.
- Robust integrations: search/extract responses are normalized before downstream use.

### 1) End-to-end deep research pipeline

The backend executes an iterative LangGraph state machine with adaptive reflection that gives the agent the ability to:

- break broad questions into independent subtasks,
- run parallel evidence gathering,
- detect missing coverage,
- and only finalize after reflection criteria are met.

### 2) Parallel subagents with adaptive search

Each subagent:

- generates multiple search queries,
- runs web search and extraction,
- evaluates source quality,
- refines queries when quality is weak,
- writes a focused partial report with evidence.

### 3) Reflection and gap-filling

Instead of stopping after one pass, the reflection node reviews progress and can create new subtasks for unresolved areas. This improves completeness and reduces blind spots.

### 4) Citation-aware finalization

After synthesis, a dedicated citation pass aligns claims to known sources and produces a citation-enriched final report.

### 5) Professional live UI

Frontend includes:

- real-time research progress,
- phase timeline,
- parallel subagent board,
- source panels and activity feeds,
- rolling AI-generated topic suggestions,
- provider/model status and selection.

### 6) How multi-agents are assigned and tracked in progress

- Assignment unit: one subagent is spawned per pending subtask (`to_run = subtasks - completed_subtasks`).
- Iteration model: after each subagent wave, reflection can append new subtasks; next wave runs only new/pending items.
- Completion tracking: successful subagent returns mark `completed_subtasks`, increment `iteration_count`, and merge/dedupe sources.
- Progress % is weighted by phase in code: `init=2`, `plan=8`, `split=5`, `scale=5`, `subagents=55`, `reflection=5`, `synthesize=12`, `cite=8`.
- UI observability comes from SSE events emitted by backend (`subagents-launch`, `subagent-step`, `subagent-search`, `subagent-sources-scored`, `subagent-extract`, `subagent-complete`, `reflection`, `complete`).


### 7) LangChain/LangGraph usage (what is actually used)

- This project uses `langgraph` (LangChain ecosystem) as the orchestration engine via `StateGraph`.
- The graph is built in `backend/graph.py`, compiled at request time, and invoked with a typed `ResearchState`.
- Role-specific LLM calls are implemented through custom provider adapters (Gemini/OpenAI/Anthropic/HuggingFace), not LangChain agent executors.
- `langsmith.traceable` is used for node/function-level observability.

### 8) Parallelism model (how agents are spun up)

- Wave-level parallelism: each pending subtask is executed concurrently via `asyncio.to_thread(...)` in `run_subagents_parallel()`.
- Per-subagent search parallelism: each subagent runs query searches in a `ThreadPoolExecutor`.
- Per-subagent extract parallelism: each subagent runs URL extraction in another `ThreadPoolExecutor`.
- Net effect: nested parallelism (subtask concurrency + I/O concurrency inside each subtask).


### 9) Gap analysis & re-assignment loop (exact behavior)

- Reflection runs after every subagent wave.
- If `iteration_count >= max_iterations`, backend forces `research_complete = true` and emits `max-iterations-reached`.
- Otherwise reflection LLM audits existing subagent reports and returns JSON with `subtasks`.
- If returned `subtasks` is non-empty: they are appended to `state["subtasks"]`; `research_complete = false`; next loop runs only newly pending subtasks.
- If returned `subtasks` is empty: `research_complete = true` and pipeline proceeds to synthesis.


---

## Screenshots

| Research in progress | Parallel subagent lanes |
|:---:|:---:|
| ![Research Live](assets/research-live.png) | ![Subagent Lanes](assets/subagents-panel.png) |

| Reflection & gap analysis | Final synthesized report |
|:---:|:---:|
| ![Reflection](assets/reflection.png) | ![Final Report](assets/final-report.png) |

---

## Tech stack

```mermaid
graph TB
    subgraph Frontend
        NX["Next.js 15"] --> RE["React 19"]
        RE --> TS["TypeScript"]
        RE --> TW["Tailwind CSS v4"]
        RE --> MD["React Markdown + GFM"]
    end

    subgraph Backend
        FA["FastAPI + Uvicorn"] --> LG["LangGraph"]
        LG --> FC["Firecrawl"]
        FA --> PD["Pydantic"]
        FA --> LS["LangSmith (optional)"]
        FA --> SB["Supabase (optional)"]
    end

    subgraph "Model Providers"
        GM["Gemini"]
        OA["OpenAI"]
        AN["Anthropic"]
        HF["HuggingFace"]
    end

    Frontend -- SSE stream --> Backend
    Backend --> GM & OA & AN & HF

    style Frontend fill:#1e1b4b,color:#c4b5fd,stroke:#6366f1
    style Backend fill:#1e1b4b,color:#93c5fd,stroke:#3b82f6
```

---

## Project structure

```
deep_research_agent/
├── backend/                # FastAPI server, graph, agents, prompts
├── frontend/               # Next.js UI
├── assets/                 # Screenshots and media (you add files)
├── requirements.txt
├── run.py                  # Backend runner
├── .env.example

```

---

## Quick start

### 1) Clone and install

```bash
git clone <your-repo-url>
cd deep_research_agent

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cd frontend
npm install
cd ..
```

### 2) Configure environment

Copy `.env.example` to `.env` and set at least:

```bash
LLM_PROVIDER=gemini
LLM_MODEL=gemini-2.5-pro
GEMINI_API_KEY=...
FIRECRAWL_API_KEY=...
```

Optional but recommended:

```bash
SUPABASE_URL=...
SUPABASE_SERVICE_KEY=...
LANGSMITH_API_KEY=...
MAX_ITERATIONS=3
QUALITY_THRESHOLD=0.7
```

### 3) Run backend

```bash
source .venv/bin/activate
python run.py
```

Backend default: `http://localhost:8000`

### 4) Run frontend

```bash
cd frontend
npm run dev
```

Frontend default: `http://localhost:3000`

---

## Configuration highlights

Global defaults:

- `LLM_PROVIDER`
- `LLM_MODEL`
- `MAX_ITERATIONS`
- `QUALITY_THRESHOLD`

Per-role model routing (examples):

- `PLANNER_PROVIDER`, `PLANNER_MODEL`
- `SUBAGENT_PROVIDER`, `SUBAGENT_MODEL`
- `COORDINATOR_PROVIDER`, `COORDINATOR_MODEL`
- `CITATION_PROVIDER`, `CITATION_MODEL`

This enables specialized model selection for planning, extraction reasoning, synthesis, and citation.

### External service setup (required vs optional)

| Service | Required | Environment variables | Purpose |
|---|---|---|---|
| LLM Provider (Gemini/OpenAI/Anthropic/HF) | Yes (at least one) | `LLM_PROVIDER`, `LLM_MODEL`, plus provider key (`GEMINI_API_KEY` or `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` or `HF_TOKEN`) | All reasoning/planning/report generation |
| Firecrawl | Yes | `FIRECRAWL_API_KEY` | Web search + extraction evidence pipeline |
| Supabase | Optional | `SUPABASE_URL`, `SUPABASE_SERVICE_KEY` | Persist checkpoints/artifacts |
| LangSmith | Optional | `LANGSMITH_API_KEY`, `LANGCHAIN_TRACING_V2`, `LANGCHAIN_PROJECT` | Trace/observability |

Notes:
- If Firecrawl is missing/unhealthy, subagent search/extract quality drops because evidence gathering depends on it.
- Supabase is best-effort persistence; pipeline still runs without it.

### Adding a new model under an existing provider

1. Quick usage (no code changes): set `LLM_MODEL=<provider-supported-model-id>` in `.env`.
2. Role-specific usage: set `<ROLE>_MODEL` (e.g., `SUBAGENT_MODEL=...`, `COORDINATOR_MODEL=...`).
3. Show model in frontend selector: add it to `AVAILABLE_MODELS` in `backend/config.py`.
4. Optional default update: add to `DEFAULT_MODELS` in `backend/config.py`.

### Adding an entirely new provider

1. Create provider adapter in `backend/providers/<new_provider>_provider.py` implementing `LLMProvider.chat(...)`.
2. Register it in `backend/providers/__init__.py` inside `get_provider()` and `list_providers()`.
3. Add provider models to `AVAILABLE_MODELS` and defaults to `DEFAULT_MODELS` in `backend/config.py`.
4. Add required API key env var in `.env.example`.
5. (Optional) set role-level overrides to route specific stages to the new provider.

---

## API overview

Key endpoints:

- `POST /api/research/stream` — starts a streaming research run (SSE)
- `POST /api/research` — starts a run and returns run_id
- `GET /api/research/{run_id}/stream` — stream by run id
- `GET /api/config` — provider/model configuration
- `GET /api/health` — backend health status
- `POST /api/topics/suggestions` — AI-generated topic cards for UI suggestions

---

## What this agent can do

- Turn broad, complex prompts into structured research plans.
- Investigate multiple perspectives in parallel.
- Gather and evaluate web evidence with quality-aware filtering.
- Detect missing information and perform follow-up research autonomously.
- Produce long-form, reasoned reports with a citation stage.
- Stream transparent progress so users can inspect the full reasoning workflow.

---

## Notes

- This system depends on external model and search APIs; output quality depends on provider health, credentials, and source availability.
- For best results, use clear research prompts with scope, time range, and target domain.

---

## References

- Anthropic Engineering: Multi-agent research system (conceptual inspiration)
    - https://www.anthropic.com/engineering/multi-agent-research-system


