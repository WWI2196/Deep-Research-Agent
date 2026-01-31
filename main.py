import asyncio
import concurrent.futures
import json
import os
import re
import time
import uuid
from typing import Any, Dict, List, Optional, TypedDict, Tuple

from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from langgraph.graph import END, StateGraph
from langsmith import traceable
from pydantic import BaseModel, Field
from supabase import create_client
from firecrawl import Firecrawl

from config import (
    PLANNER_CONFIG,
    TASK_SPLITTER_CONFIG,
    SCALING_CONFIG,
    SUBAGENT_CONFIG,
    COORDINATOR_CONFIG,
    REFLECTION_CONFIG,
    CITATION_CONFIG,
)

load_dotenv()

FIRECRAWL_MAX_RETRIES = int(os.environ.get("FIRECRAWL_MAX_RETRIES", "3"))
FIRECRAWL_BACKOFF_BASE = float(os.environ.get("FIRECRAWL_BACKOFF_BASE", "0.5"))

PLANNER_SYSTEM_INSTRUCTIONS = """
You will be given a research task by a user. Your job is to produce a set of
instructions for a researcher that will complete the task. Do NOT complete the
task yourself, just provide instructions on how to complete it.

GUIDELINES:
1. Maximize specificity and detail. Include all known user preferences and
   explicitly list key attributes or dimensions to consider.
2. If essential attributes are missing, explicitly state that they are open-ended.
3. Avoid unwarranted assumptions. Treat unspecified dimensions as flexible.
4. Use the first person (from the user's perspective).
5. When helpful, explicitly ask the researcher to include tables.
6. Include the expected output format (e.g. structured report with headers).
7. Preserve the input language unless the user explicitly asks otherwise.
8. Sources: prefer primary / official / original sources.
"""

TASK_SPLITTER_SYSTEM_INSTRUCTIONS = """
You will be given a set of research instructions (a research plan).
Your job is to break this plan into a set of coherent, non-overlapping
subtasks that can be researched independently by separate agents.

Requirements:
- For the number of subtasks, use your judgment (3 - 12 is ideal).
- The number of subtasks should cover the full scope of the research plan.
- Each subtask should have:
  - an 'id' (short string),
  - a 'title' (short descriptive title),
  - a 'description' (clear, detailed instructions for the sub-agent),
  - 'objective' (what the subagent should accomplish),
  - 'output_format' (expected structure of the report),
  - 'tool_guidance' (which tools to prioritize: web search, extract, etc.),
  - 'source_types' (preferred sources: academic, official, news, etc.),
  - 'boundaries' (what NOT to include to avoid overlap).
- Subtasks should collectively cover the full scope of the original plan
  without unnecessary duplication.
- Prefer grouping by dimensions: time periods, regions, actors, themes,
  causal mechanisms, evidence types, and source categories.
- Each description should be very clear and detailed about everything that
  the agent needs to research to cover that topic.
- Do not include a final task that will put everything together.

Output format:
Return ONLY valid JSON with this schema:
{
  "subtasks": [
    {
      "id": "string",
      "title": "string",
      "description": "string",
      "objective": "string",
      "output_format": "string",
      "tool_guidance": "string",
      "source_types": "string",
      "boundaries": "string"
    }
  ]
}
"""

SCALING_RULES_SYSTEM_INSTRUCTIONS = """
You will be given a user query and a research plan.
Your job is to estimate the research complexity and return a resource plan.

Rules:
- simple: 1 subagent, 3-10 tool calls, 3-5 sources total
- moderate: 2-4 subagents, 10-15 tool calls each, 8-15 sources total
- complex: 6-12 subagents, 15-25 tool calls each, 15-40 sources total

Return JSON with:
{
  "complexity": "simple|moderate|complex",
  "subagent_count": number,
  "tool_calls_per_subagent": number,
  "target_sources": number
}
"""

REFLECTION_INSTRUCTIONS = """
You are a research coordinator.
Your job is to review the current research progress and determine if more information is needed.

Context:
- User Query: {user_query}
- Research Plan: {research_plan}
- Past Subtasks: {past_subtasks}
- Subagent Reports:
{subagent_reports}

Analyze the reports. Identify any:
1. Missing information required by the original plan.
2. Knowledge gaps where the agents failed to find sufficient data.
3. Contradictions between reports that need resolution.

If the research is sufficient to answer the User Query comprehensively with high density information, return an empty list of subtasks.

If more research is needed, generate a list of NEW subtasks to address these specific gaps.
Do NOT regenerate subtasks that have already been completed.
Keep the new subtasks focused and specific.

Output valid JSON matching this schema:
{
  "subtasks": [
    {
      "id": "new_id_1",
      "title": "Title",
      "description": "Detailed instructions...",
      "objective": "Objective...",
      "output_format": "markdown",
      "tool_guidance": "search strategies...",
      "source_types": "preferred sources...",
      "boundaries": "what to exclude..."
    }
  ]
}
"""

SUBAGENT_REPORT_PROMPT = """
You are a specialized research sub-agent.

Global user query:
{user_query}

Overall research plan:
{research_plan}

Your specific subtask (ID: {subtask_id}, Title: {subtask_title}) is:
<<<SUBTASK>>>
Description: {subtask_description}
Objective: {subtask_objective}
Output Format: {subtask_output_format}
Tool Guidance: {subtask_tool_guidance}
Preferred Sources: {subtask_source_types}
Boundaries (do NOT include): {subtask_boundaries}
<<<END SUBTASK>>>

IMPORTANT INSTRUCTIONS:
1. Focus on "distilled insights" and high information density.
2. Avoid verbose summaries; prioritize facts, data, and unique findings.
3. Start with BROAD search queries, then narrow down based on results.
4. Prefer PRIMARY sources (official docs, academic papers) over SEO content.
5. If you find conflicting information, note it explicitly.

You will be given extracted evidence snippets and sources.
Produce a MARKDOWN report with this structure:

# [{subtask_id}] {subtask_title}

## Key Findings (Distilled)
- High-density bullet points with clear facts/numbers.

## Detailed Analysis
- In-depth exploration of the subtask topic.
- Connection of evidence to the objective.

## Source Quality
- Assessment of source reliability.

## Sources
- [Title](url) - source type and relevance
"""

SYNTHESIS_PROMPT = """
You are the LEAD RESEARCH COORDINATOR AGENT.

User query:
<<<USER QUERY>>>
{user_query}
<<<END USER QUERY>>>

Research plan:
<<<RESEARCH PLAN>>>
{research_plan}
<<<END RESEARCH PLAN>>>

Subagent reports:
{subagent_reports}

Your job: synthesize a single, coherent, deeply researched report.

Final report requirements:
- Integrate all sub-agent findings; avoid redundancy.
- Make the structure clear with headings and subheadings.
- Highlight key drivers/mechanisms, temporal evolution, geographic patterns,
  socioeconomic correlates, and uncertainties.
- End with:
  - Open Questions and Further Research
  - Bibliography / Sources (deduplicated)
"""

CITATION_PROMPT = """
You are a citation agent. Insert citations for claims in the report using the
provided sources and evidence snippets. Preserve the report content and add
citations inline using [^n] markers with a numbered bibliography.

Report:
<<<REPORT>>>
{report}
<<<END REPORT>>>

Sources and snippets:
{sources}

Return the fully cited report.
"""


class Subtask(BaseModel):
    id: str = Field(..., description="Short identifier for the subtask.")
    title: str = Field(..., description="Short descriptive title of the subtask.")
    description: str = Field(
        ..., description="Clear, detailed instructions for the sub-agent."
    )
    objective: str = Field(default="", description="What the subagent should accomplish.")
    output_format: str = Field(default="", description="Expected structure of output.")
    tool_guidance: str = Field(default="", description="Which tools to prioritize.")
    source_types: str = Field(default="", description="Preferred source types.")
    boundaries: str = Field(default="", description="What to exclude to avoid overlap.")


class SubtaskList(BaseModel):
    subtasks: List[Subtask] = Field(
        ..., description="List of subtasks that together cover the whole plan."
    )


class ScalingPlan(BaseModel):
    complexity: str
    subagent_count: int
    tool_calls_per_subagent: int
    target_sources: int


class ResearchState(TypedDict, total=False):
    run_id: str
    user_query: str
    research_plan: str
    subtasks: List[Dict[str, Any]]
    scaling: Dict[str, Any]
    subagent_reports: List[str]
    sources: List[Dict[str, Any]]
    report: str
    cited_report: str
    events: List[Dict[str, Any]]
    errors: List[str]
    # New fields for iterative research
    iteration_count: int
    completed_subtasks: List[str]
    max_iterations: int
    research_complete: bool
    memory: Dict[str, Any]  # External memory for context preservation
    quality_threshold: float
    current_quality_score: float


def _extract_json(text: str) -> str:
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return text
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return ""


def _clean_deepseek_think(content: str) -> str:
    """
    Removes <think>...</think> tags from DeepSeek R1 output.
    """
    if "<think>" in content and "</think>" in content:
        # Remove content between tags, including tags
        # DOTALL is needed to match newlines within the think block
        cleaned = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
        return cleaned.strip()
    return content


def _get_inference_client(provider: str = "auto") -> InferenceClient:
    return InferenceClient(
        api_key=os.environ.get("HF_TOKEN", ""),
        provider=provider or os.environ.get("HF_PROVIDER", "auto"),
    )


def _hf_chat(
    model_id: str,
    messages: List[Dict[str, str]],
    response_format: Optional[Dict[str, Any]] = None,
    temperature: float = 0.2,
    provider: str = "auto",
) -> str:
    client = _get_inference_client(provider)
    completion = client.chat.completions.create(
        model=model_id,
        messages=messages,
        response_format=response_format,
        temperature=temperature,
    )
    content = completion.choices[0].message.content or ""
    return _clean_deepseek_think(content)


def _get_supabase_client():
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not url or not key:
        return None
    return create_client(url, key)


def _retry_call(func, *args, **kwargs):
    last_error = None
    for attempt in range(FIRECRAWL_MAX_RETRIES):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            last_error = exc
            sleep_for = FIRECRAWL_BACKOFF_BASE * (2**attempt)
            time.sleep(sleep_for)
    raise last_error


def persist_state(state: ResearchState, phase: str) -> None:
    client = _get_supabase_client()
    if not client:
        return

    try:
        checkpoint_payload = {
            "run_id": state.get("run_id"),
            "phase": phase,
            "state": state,
            "created_at": int(time.time()),
        }
        client.table("deep_research_checkpoints").insert(checkpoint_payload).execute()
    except Exception:
        try:
            fallback_payload = {
                "run_id": state.get("run_id"),
                "phase": phase,
                "state": state,
                "updated_at": int(time.time()),
            }
            client.table("deep_research_runs").upsert(
                fallback_payload, on_conflict="run_id"
            ).execute()
        except Exception:
            pass


def persist_artifact(
    run_id: str,
    artifact_type: str,
    content: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    client = _get_supabase_client()
    if not client:
        return

    payload = {
        "run_id": run_id,
        "artifact_type": artifact_type,
        "content": content,
        "metadata": metadata or {},
        "created_at": int(time.time()),
    }

    try:
        client.table("deep_research_artifacts").insert(payload).execute()
    except Exception:
        pass


def _log_event(state: ResearchState, message: str, event_type: str) -> None:
    events = state.get("events", [])
    events.append(
        {
            "type": event_type,
            "message": message,
            "timestamp": int(time.time()),
        }
    )
    state["events"] = events


@traceable(name="plan_research")
def generate_research_plan(user_query: str) -> str:
    return _hf_chat(
        model_id=PLANNER_CONFIG.model_id,
        messages=[
            {"role": "system", "content": PLANNER_SYSTEM_INSTRUCTIONS},
            {"role": "user", "content": user_query},
        ],
        provider=PLANNER_CONFIG.provider,
    )


@traceable(name="split_subtasks")
def split_into_subtasks(research_plan: str) -> List[Dict[str, Any]]:
    response = _hf_chat(
        model_id=TASK_SPLITTER_CONFIG.model_id,
        messages=[
            {"role": "system", "content": TASK_SPLITTER_SYSTEM_INSTRUCTIONS},
            {"role": "user", "content": research_plan},
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "subtaskList",
                "schema": SubtaskList.model_json_schema(),
                "strict": True,
            },
        },
        temperature=0.2,
        provider=TASK_SPLITTER_CONFIG.provider,
    )

    content = response.strip()
    if not content:
        raise ValueError("Empty model response from task splitter.")

    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        extracted = _extract_json(content)
        if not extracted:
            raise
        payload = json.loads(extracted)

    return payload["subtasks"]


@traceable(name="compute_scaling")
def compute_scaling(user_query: str, research_plan: str) -> Dict[str, Any]:
    response = _hf_chat(
        model_id=SCALING_CONFIG.model_id,
        messages=[
            {"role": "system", "content": SCALING_RULES_SYSTEM_INSTRUCTIONS},
            {
                "role": "user",
                "content": f"Query: {user_query}\n\nPlan:\n{research_plan}",
            },
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "scalingPlan",
                "schema": ScalingPlan.model_json_schema(),
                "strict": True,
            },
        },
        temperature=0.1,
        provider=SCALING_CONFIG.provider,
    )

    content = response.strip()
    if not content:
        raise ValueError("Empty model response from scaling planner.")

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        extracted = _extract_json(content)
        if not extracted:
            raise
        return json.loads(extracted)


@traceable(name="generate_search_queries")
def generate_search_queries(subtask: Dict[str, Any]) -> List[str]:
    prompt = (
        "Generate 3-6 web search queries for the subtask.\n"
        "IMPORTANT: Start with SHORT, BROAD queries to explore the landscape.\n"
        "Avoid overly specific or long queries that return few results.\n"
        "Return as JSON: {\"queries\": [\"...\"]}.\n\n"
        f"Subtask: {subtask['title']}\n"
        f"Description: {subtask['description']}\n"
        f"Objective: {subtask.get('objective', '')}\n"
        f"Preferred Sources: {subtask.get('source_types', '')}"
    )
    response = _hf_chat(
        model_id=SUBAGENT_CONFIG.model_id,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        provider=SUBAGENT_CONFIG.provider,
    )
    try:
        payload = json.loads(_extract_json(response))
        return payload.get("queries", [])
    except Exception:
        return [subtask["title"]]


def _get_firecrawl() -> Firecrawl:
    return Firecrawl(api_key=os.environ.get("FIRECRAWL_API_KEY", ""))


def _score_source_quality(source: Dict[str, Any]) -> float:
    """Score source quality based on URL, title, and type.
    Prefer primary sources over SEO-optimized content farms."""
    url = source.get("url", "").lower()
    title = source.get("title", "").lower()
    score = 0.5  # baseline
    
    # Primary/authoritative sources (higher score)
    primary_indicators = [
        ".gov", ".edu", ".org", "arxiv", "doi.org", "pubmed",
        "official", "whitepaper", "academic", "journal",
        "nature.com", "science.org", "ieee"
    ]
    for indicator in primary_indicators:
        if indicator in url or indicator in title:
            score += 0.3
            break
    
    # SEO content farms / low quality (lower score)
    low_quality = [
        "listicle", "top 10", "you won't believe",
        "clickbait", "viral", "trending"
    ]
    for indicator in low_quality:
        if indicator in title:
            score -= 0.3
            break
    
    return max(0.0, min(1.0, score))


@traceable(name="run_subagent")
def run_subagent(
    user_query: str,
    research_plan: str,
    subtask: Dict[str, Any],
    tool_budget: int,
) -> Dict[str, Any]:
    firecrawl = _get_firecrawl()
    queries = generate_search_queries(subtask)

    sources: List[Dict[str, Any]] = []
    evidence: List[Dict[str, Any]] = []

    # Parallel search execution for multiple queries
    search_queries = queries[: max(1, min(6, tool_budget))]

    def perform_search(query):
        try:
            return _retry_call(firecrawl.search, query=query, limit=5)
        except Exception:
            return None

    with concurrent.futures.ThreadPoolExecutor() as executor:
        search_results_list = list(executor.map(perform_search, search_queries))

    for search_results in search_results_list:
        if not search_results or not search_results.get("data"):
            continue
        for item in search_results["data"]:
            source = {
                "title": item.get("title"),
                "url": item.get("url"),
                "description": item.get("description"),
                "source": "search",
            }
            source["quality_score"] = _score_source_quality(source)
            sources.append(source)

    # Sort sources by quality score and deduplicate
    sources.sort(key=lambda x: x.get("quality_score", 0), reverse=True)
    unique_urls = []
    filtered_sources = []
    for source in sources:
        url = source.get("url")
        if url and url not in unique_urls:
            unique_urls.append(url)
            filtered_sources.append(source)

    top_urls = unique_urls[: min(8, tool_budget)]

    # Extract content from top-quality sources
    def perform_extract(url):
        try:
            return url, _retry_call(
                firecrawl.extract,
                [url],
                {
                    "prompt": (
                        f"Extract key facts, data, expert statements, and primary evidence about: {subtask['title']}. "
                        f"Prioritize factual information from authoritative sources."
                    )
                },
            )
        except Exception:
            return url, None

    with concurrent.futures.ThreadPoolExecutor() as executor:
        extract_results = list(executor.map(perform_extract, top_urls))

    for url, extract_result in extract_results:
        if extract_result and extract_result.get("success"):
            data = extract_result.get("data")
            evidence.append({"url": url, "data": data})

    evidence_text = "\n\n".join(
        [f"[From {e['url']}]: {e['data']}" for e in evidence]
    )

    # Generate report with enhanced prompt
    report = _hf_chat(
        model_id=SUBAGENT_CONFIG.model_id,
        messages=[
            {
                "role": "system",
                "content": SUBAGENT_REPORT_PROMPT.format(
                    user_query=user_query,
                    research_plan=research_plan,
                    subtask_id=subtask["id"],
                    subtask_title=subtask["title"],
                    subtask_description=subtask["description"],
                    subtask_objective=subtask.get("objective", ""),
                    subtask_output_format=subtask.get("output_format", ""),
                    subtask_tool_guidance=subtask.get("tool_guidance", ""),
                    subtask_source_types=subtask.get("source_types", ""),
                    subtask_boundaries=subtask.get("boundaries", ""),
                ),
            },
            {"role": "user", "content": f"Evidence:\n{evidence_text}"},
        ],
        temperature=0.2,
        provider=SUBAGENT_CONFIG.provider,
    )

    return {
        "subtask_id": subtask["id"],
        "report": report,
        "sources": filtered_sources[:20],  # Keep top quality sources
        "evidence_count": len(evidence),
    }


@traceable(name="synthesize_report")
def synthesize_report(user_query: str, research_plan: str, reports: List[str]) -> str:
    return _hf_chat(
        model_id=COORDINATOR_CONFIG.model_id,
        messages=[
            {
                "role": "system",
                "content": SYNTHESIS_PROMPT.format(
                    user_query=user_query,
                    research_plan=research_plan,
                    subagent_reports="\n\n".join(reports),
                ),
            }
        ],
        temperature=0.2,
        provider=COORDINATOR_CONFIG.provider,
    )


@traceable(name="citation_agent")
def add_citations(report: str, sources: List[Dict[str, Any]]) -> str:
    sources_text = "\n".join(
        [
            f"- {s.get('title', 'Source')} | {s.get('url')} | {s.get('description', '')}"
            for s in sources
            if s.get("url")
        ]
    )
    return _hf_chat(
        model_id=CITATION_CONFIG.model_id,
        messages=[
            {
                "role": "system",
                "content": CITATION_PROMPT.format(report=report, sources=sources_text),
            }
        ],
        temperature=0.1,
        provider=CITATION_CONFIG.provider,
    )


async def run_subagents_parallel(
    user_query: str,
    research_plan: str,
    subtasks: List[Dict[str, Any]],
    tool_calls_per_subagent: int,
) -> Dict[str, Any]:
    """Run subagents in parallel with error handling for individual failures."""
    tasks = [
        asyncio.to_thread(
            run_subagent,
            user_query,
            research_plan,
            subtask,
            tool_calls_per_subagent,
        )
        for subtask in subtasks
    ]
    
    # Gather with exception handling - don't fail entire batch if one fails
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    reports = []
    sources = []
    successful_results = []
    
    for r in results:
        if isinstance(r, Exception):
            # Log error but continue with other results
            continue
        reports.append(r["report"])
        sources.extend(r.get("sources", []))
        successful_results.append(r)
    
    # Deduplicate sources by URL
    seen_urls = set()
    unique_sources = []
    for source in sources:
        url = source.get("url")
        if url and url not in seen_urls:
            seen_urls.add(url)
            unique_sources.append(source)
    
    return {
        "reports": reports,
        "sources": unique_sources,
        "raw": successful_results,
        "success_count": len(successful_results),
        "total_count": len(subtasks),
    }


def _evaluate_research_quality(state: ResearchState) -> float:
    """Evaluate if research is sufficient to answer the query.
    Returns score 0.0-1.0 indicating completeness."""
    
    # Simple heuristics for quality assessment
    reports = state.get("subagent_reports", [])
    sources = state.get("sources", [])
    
    if not reports:
        return 0.0
    
    score = 0.0
    
    # Factor 1: Number of successful reports
    report_coverage = len(reports) / max(1, len(state.get("subtasks", [])))
    score += report_coverage * 0.4
    
    # Factor 2: Source count and quality
    if sources:
        avg_quality = sum(s.get("quality_score", 0.5) for s in sources) / len(sources)
        source_score = min(1.0, len(sources) / 10.0) * avg_quality
        score += source_score * 0.4
    
    # Factor 3: Report length (proxy for depth)
    total_length = sum(len(r) for r in reports)
    length_score = min(1.0, total_length / 5000)
    score += length_score * 0.2
    
    return min(1.0, score)


def build_graph() -> StateGraph:
    graph = StateGraph(ResearchState)

    @traceable(name="init_state")
    def init_state(state: ResearchState) -> ResearchState:
        state["run_id"] = state.get("run_id") or str(uuid.uuid4())
        state["events"] = state.get("events", [])
        state["errors"] = state.get("errors", [])
        state["iteration_count"] = 0
        state["max_iterations"] = int(os.environ.get("MAX_RESEARCH_ITERATIONS", "2"))
        state["research_complete"] = False
        state["quality_threshold"] = float(os.environ.get("QUALITY_THRESHOLD", "0.7"))
        state["current_quality_score"] = 0.0
        state["memory"] = {}  # External memory for context preservation
        persist_state(state, "init")
        return state

    @traceable(name="plan_node")
    def plan_node(state: ResearchState) -> ResearchState:
        _log_event(state, "Generating research plan", "plan")
        state["research_plan"] = generate_research_plan(state["user_query"])
        persist_state(state, "plan")
        return state

    @traceable(name="split_node")
    def split_node(state: ResearchState) -> ResearchState:
        _log_event(state, "Splitting into subtasks", "split")
        state["subtasks"] = split_into_subtasks(state["research_plan"])
        persist_state(state, "split")
        return state

    @traceable(name="scale_node")
    def scale_node(state: ResearchState) -> ResearchState:
        _log_event(state, "Computing scaling plan", "scale")
        state["scaling"] = compute_scaling(state["user_query"], state["research_plan"])
        persist_state(state, "scale")
        return state

    @traceable(name="subagents_node")
    def subagents_node(state: ResearchState) -> ResearchState:
        iteration = state.get("iteration_count", 0)
        completed = set(state.get("completed_subtasks", []))
        
        # Filter subtasks that haven't been completed yet
        all_subtasks = state.get("subtasks", [])
        subtasks_to_run = [s for s in all_subtasks if s["id"] not in completed]
        
        if not subtasks_to_run:
            _log_event(state, "No new subtasks to run.", "subagents")
            return state

        _log_event(state, f"Running {len(subtasks_to_run)} subagents (iteration {iteration + 1})", "subagents")
        
        # Store research plan in memory to prevent loss
        state["memory"]["research_plan"] = state["research_plan"]
        state["memory"]["subtasks"] = state["subtasks"]
        
        tool_budget = state.get("scaling", {}).get("tool_calls_per_subagent", 10)
        results = asyncio.run(
            run_subagents_parallel(
                state["user_query"],
                state["research_plan"],
                subtasks_to_run,
                tool_budget,
            )
        )
        
        # Merge with existing reports if this is a follow-up iteration
        existing_reports = state.get("subagent_reports", [])
        existing_sources = state.get("sources", [])
        
        state["subagent_reports"] = existing_reports + results["reports"]
        
        # Update completed subtasks
        new_completed = [item["subtask_id"] for item in results.get("raw", [])]
        state["completed_subtasks"] = list(completed.union(new_completed))

        # Merge and deduplicate sources
        all_sources = existing_sources + results["sources"]
        seen_urls = set()
        unique_sources = []
        for s in all_sources:
            url = s.get("url")
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_sources.append(s)
        state["sources"] = unique_sources
        
        for item in results.get("raw", []):
            persist_artifact(
                state.get("run_id", ""),
                "subagent_report",
                item.get("report", ""),
                {
                    "subtask_id": item.get("subtask_id"),
                    "iteration": iteration,
                },
            )
        
        state["iteration_count"] = iteration + 1
        
        persist_state(state, f"subagents_iter_{iteration}")
        return state

    @traceable(name="synthesize_node")
    def synthesize_node(state: ResearchState) -> ResearchState:
        _log_event(state, "Synthesizing final report", "synthesis")
        state["report"] = synthesize_report(
            state["user_query"], state["research_plan"], state["subagent_reports"]
        )
        persist_artifact(
            state.get("run_id", ""),
            "final_report",
            state["report"],
            {},
        )
        persist_state(state, "synthesis")
        return state

    @traceable(name="citation_node")
    def citation_node(state: ResearchState) -> ResearchState:
        _log_event(state, "Adding citations", "citation")
        state["cited_report"] = add_citations(
            state["report"], state.get("sources", [])
        )
        persist_artifact(
            state.get("run_id", ""),
            "cited_report",
            state["cited_report"],
            {},
        )
        persist_state(state, "citation")
        return state

    @traceable(name="reflection_node")
    def reflection_node(state: ResearchState) -> ResearchState:
        """Reflect on research depth and identify gaps."""
        current_subtasks = state.get("subtasks", [])
        reports = state.get("subagent_reports", [])
        iteration = state.get("iteration_count", 0)
        max_iter = state.get("max_iterations", 2)
        
        # Format past subtasks (ID/Title) for context
        past_subtasks = ", ".join([f"{s.get('id')}: {s.get('title')}" for s in current_subtasks])
        
        if iteration >= max_iter:
             state["research_complete"] = True
             _log_event(state, "Max iterations reached. Proceeding to synthesis.", "reflection")
             return state

        _log_event(state, "Reflecting on research progress...", "reflection")
        
        response = _hf_chat(
            model_id=REFLECTION_CONFIG.model_id,
            messages=[
                {
                    "role": "user", 
                    "content": REFLECTION_INSTRUCTIONS.format(
                        user_query=state["user_query"],
                        research_plan=state["research_plan"],
                        past_subtasks=past_subtasks,
                        subagent_reports="\n\n".join(reports)
                    )
                }
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "subtaskList",
                    "schema": SubtaskList.model_json_schema(),
                    "strict": True,
                },
            },
            provider=REFLECTION_CONFIG.provider,
        )
        
        # Parse response
        content = response.strip()
        new_subtasks = []
        try:
             payload = json.loads(content)
             new_subtasks = payload.get("subtasks", [])
        except json.JSONDecodeError:
            extracted = _extract_json(content)
            if extracted:
                 try:
                    payload = json.loads(extracted)
                    new_subtasks = payload.get("subtasks", [])
                 except:
                    pass
        
        if new_subtasks:
            _log_event(state, f"Reflection identified {len(new_subtasks)} new subtasks: {[s['title'] for s in new_subtasks]}", "reflection")
            state["subtasks"].extend(new_subtasks)
            state["research_complete"] = False
        else:
            _log_event(state, "Reflection identified no new gaps. Research complete.", "reflection")
            state["research_complete"] = True
            
        persist_state(state, "reflection")
        return state
    
    def should_continue_research(state: ResearchState) -> str:
        """Conditional edge: continue research or move to synthesis."""
        if state.get("research_complete", False):
            return "synthesize"
        return "subagents"

    graph.add_node("init", init_state)
    graph.add_node("plan", plan_node)
    graph.add_node("split", split_node)
    graph.add_node("scale", scale_node)
    graph.add_node("subagents", subagents_node)
    graph.add_node("reflection", reflection_node)
    graph.add_node("synthesize", synthesize_node)
    graph.add_node("cite", citation_node)

    graph.set_entry_point("init")
    graph.add_edge("init", "plan")
    graph.add_edge("plan", "split")
    graph.add_edge("split", "scale")
    graph.add_edge("scale", "subagents")
    graph.add_edge("subagents", "reflection")
    
    # Conditional edge: either continue research or synthesize
    graph.add_conditional_edges(
        "reflection",
        should_continue_research,
        {
            "synthesize": "synthesize",
            "subagents": "subagents",
        }
    )
    
    graph.add_edge("synthesize", "cite")
    graph.add_edge("cite", END)

    return graph


def run_deep_research(user_query: str) -> str:
    graph = build_graph().compile()
    final_state = graph.invoke({"user_query": user_query})
    return final_state.get("cited_report") or final_state.get("report") or ""


if __name__ == "__main__":
    query = "research the climate in northern france"
    print(run_deep_research(query))
