# Deep Research Agent - Comprehensive Analysis & Recommendations

## Executive Summary

After analyzing your current implementation, Anthropic's multi-agent research system, the open-deep-research repository, and available technologies, here's a detailed breakdown of improvements and recommendations.

---

## 1. MEMORY PERSISTENCE - Technology Recommendations

### Current State
- No persistent memory across sessions
- Research plan and subtasks regenerated each time
- No checkpoint/resume capability

### **Recommended Solution: LangGraph Checkpointing + PostgreSQL**

#### Why This Stack?

**LangGraph Checkpointing**
- ✅ Built-in state management and checkpointing
- ✅ Native thread support for conversation continuity
- ✅ Automatic serialization/deserialization
- ✅ Support for resumption after failures
- ✅ Time-travel capabilities (replay from any checkpoint)
- ✅ Async support for better performance

**PostgreSQL Storage**
- ✅ Production-ready and reliable
- ✅ ACID compliance for data integrity
- ✅ Excellent performance for concurrent access
- ✅ You already have `POSTGRES_URL` support based on open-deep-research
- ✅ Rich query capabilities for analytics
- ✅ Horizontal scaling options

#### Implementation Pattern

```python
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool

# Setup connection pool
pool = ConnectionPool(
    conninfo=os.environ["POSTGRES_URL"],
    max_size=20,
)

# Create checkpointer
checkpointer = PostgresSaver(pool)
checkpointer.setup()  # Creates necessary tables

# Use with config
config = {
    "configurable": {
        "thread_id": "research_session_123",
        "checkpoint_ns": "research",
    }
}
```

#### Alternative: **Redis + PostgreSQL Hybrid**

For even better performance in production:
- **Redis**: In-memory caching for active research sessions
- **PostgreSQL**: Long-term persistence and history
- **Pattern**: Write to both, read from Redis first (cache-aside)

```python
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.memory import InMemoryStore
from langchain.embeddings import init_embeddings

# Persistent checkpointing
checkpointer = PostgresSaver.from_conn_string(os.environ["POSTGRES_URL"])

# In-memory store for cross-thread data (e.g., user preferences)
memory_store = InMemoryStore(
    index={
        "embed": init_embeddings("openai:text-embedding-3-small"),
        "dims": 1536,
        "fields": ["$"]  # Embed all fields
    }
)
```

---

## 2. FIRECRAWL MCP CAPABILITIES

### Available Tools (from docs.firecrawl.dev/mcp)

**Currently Used:**
- `search` - Web search
- `scrapeUrl` - Single URL scraping
- `extract` - Content extraction with prompts

**Not Utilized But Available:**
```python
# Additional Firecrawl capabilities you should add:
1. Batch scraping - scrape multiple URLs in parallel
2. Crawl - Deep website crawling
3. Map - Get sitemap of a website
4. Screenshot - Visual capture of pages
```

### Recommendations from Open-Deep-Research

The `open-deep-research` repo shows advanced patterns:

**1. Iterative Analysis Loop**
```python
# They use a continuous feedback loop:
while currentDepth < maxDepth:
    - Search → Extract → Analyze → Plan next steps
    - Reasoning model decides if more research needed
    - Time-based cutoff (4.5 minutes)
```

**2. Activity Tracking**
```python
# Real-time progress updates
addActivity({
    type: 'search' | 'extract' | 'analyze' | 'reasoning' | 'synthesis',
    status: 'pending' | 'complete' | 'error',
    message: string,
    timestamp: string,
    depth: number,
})
```

**3. Source Management**
```python
# Track and display sources with relevance
addSource({
    url: string,
    title: string,
    description: string,
    relevance: number  # 1.0 - (index * 0.1)
})
```

---

## 3. CITATION AGENT - Integration Plan

### Design Pattern (Anthropic Approach)

**Separate Agent at the End**
```python
def create_citation_agent():
    """
    Runs AFTER all research is complete.
    Input: Final report + All sources
    Output: Report with inline citations
    """
    
    CITATION_PROMPT = """
    You are a citation specialist. Given a research report and a list of sources,
    add proper citations to all claims.
    
    RULES:
    1. Use [1], [2], etc. for inline citations
    2. Match claims to specific sources
    3. Add "Sources" section at the end
    4. Flag uncited claims
    
    Report: {report}
    Sources: {sources}
    """
    
    citation_model = InferenceClientModel(
        model_id="Qwen/Qwen3-Next-80B-A3B-Thinking",  # Good for structured output
        api_key=os.environ["HF_TOKEN"],
        provider="auto",
    )
    
    return ToolCallingAgent(
        tools=[],
        model=citation_model,
        add_base_tools=False,
        name="citation_agent"
    )
```

### Integration into Your Flow

```python
def run_deep_research(user_query: str) -> str:
    # ... existing research flow ...
    
    final_report = coordinator.run(coordinator_prompt)
    
    # NEW: Add citations
    citation_agent = create_citation_agent()
    
    # Collect all sources from subagents
    all_sources = []  # Populated during research
    
    cited_report = citation_agent.run(
        CITATION_PROMPT.format(
            report=final_report,
            sources=json.dumps(all_sources, indent=2)
        )
    )
    
    return cited_report
```

---

## 4. EXTENDED THINKING MODE

### Current Usage
You're ONLY using thinking mode in `generate_research_plan` with Qwen3-Next-80B-A3B-Thinking.

### Recommended Expansion

**Research Plan Generation** ✅ (Already doing this)
- Model: Qwen3-Next-80B-A3B-Thinking
- Purpose: Strategic planning

**Subtask Analysis** ⚠️ (Should add)
```python
# In split_into_subtasks, add thinking
MODEL_ID = "Qwen/Qwen3-Next-80B-A3B-Thinking"  # Use thinking model
# This will improve subtask decomposition quality
```

**Coordinator Decision-Making** ⚠️ (Should add)
```python
# Add extended thinking for the coordinator
# When deciding if research is complete or needs more depth
```

**Subagent Search Strategy** ⚠️ (Should add thinking between iterations)
```python
SUBAGENT_WITH_THINKING_TEMPLATE = """
Before each search iteration:
<thinking>
- What have I learned so far?
- What gaps remain?
- What's the best next query?
</thinking>

Then execute the search.
"""
```

### Model Selection for Thinking

**For Planning & Strategy:**
- `Qwen/Qwen3-Next-80B-A3B-Thinking` ✅ (Good choice)
- Alternative: `deepseek-ai/DeepSeek-R1` (open-source reasoning)

**For Execution (non-thinking):**
- `MiniMaxAI/MiniMax-M1-80k` ✅ (Good - large context window)
- Keep using this for subagents and coordinator

---

## 5. BETTER TASK DECOMPOSITION

### Current Issues

Your current approach is good but can be enhanced:

```python
# Current: Generic decomposition
subtasks = split_into_subtasks(research_plan)
```

### Anthropic's Approach - More Detailed Instructions

```python
IMPROVED_TASK_SPLITTER_PROMPT = """
You will decompose a research plan into subtasks.

CRITICAL REQUIREMENTS:

1. **Explicit Division of Labor**
   - Each subtask must have ZERO overlap
   - Specify exactly what to search for
   - Define exclusions (what NOT to cover)
   
2. **Detailed Subtask Descriptions**
   Include for each:
   - Primary focus
   - Specific questions to answer
   - Expected sources (academic, news, official, etc.)
   - Time period if relevant
   - Geographic scope if relevant
   - Key entities/concepts to research
   
3. **Quality over Quantity**
   - Prefer 3-8 well-defined subtasks over 12 vague ones
   - Each should be independently researchable
   - Together they must cover 100% of the research plan

EXAMPLE GOOD SUBTASK:
{{
  "id": "climate_trends_1990_2020",
  "title": "Historical Climate Trends (1990-2020)",
  "description": "Research historical climate data for Northern France covering 
  the 30-year period from 1990-2020. Focus on:
  - Temperature trends (annual averages, seasonal variations)
  - Precipitation patterns (rainfall, snowfall)
  - Extreme weather events (heatwaves, storms, floods)
  - Data sources: Météo-France official records, academic climate studies
  - DO NOT cover future projections (separate subtask)
  - DO NOT cover other regions (focus only on Hauts-de-France, Normandy, Grand Est)"
}}

EXAMPLE BAD SUBTASK:
{{
  "id": "climate",
  "title": "Climate Info",
  "description": "Research climate information"
}}
"""
```

### Add Subtask Validation

```python
def validate_subtasks(subtasks: List[dict], research_plan: str) -> List[dict]:
    """
    Use an LLM to validate subtask quality before execution.
    """
    
    VALIDATION_PROMPT = """
    Review these subtasks for a research project.
    
    Research Plan: {research_plan}
    Subtasks: {subtasks}
    
    Check for:
    1. Coverage: Do they cover the full research plan?
    2. Overlap: Is there duplication?
    3. Clarity: Are instructions specific enough?
    
    Return:
    - "approved": if good
    - "revise": with specific improvement suggestions
    """
    
    # Use a reasoning model to validate
    validation_client = InferenceClient(
        model_id="Qwen/Qwen3-Next-80B-A3B-Thinking",
        api_key=os.environ["HF_TOKEN"],
    )
    
    result = validation_client.chat.completions.create(...)
    
    if result.status == "revise":
        # Regenerate with feedback
        return split_into_subtasks(research_plan, feedback=result.suggestions)
    
    return subtasks
```

---

## 6. SCALING RULES IMPLEMENTATION

### Anthropic's Guidelines

```
Simple fact-finding: 1 agent, 3-10 tool calls
Direct comparisons: 2-4 subagents, 10-15 calls each
Complex research: 10+ subagents, clearly divided
```

### Implementation

```python
def determine_research_complexity(user_query: str, research_plan: str) -> dict:
    """
    Analyze query to determine appropriate research scale.
    """
    
    COMPLEXITY_ANALYSIS_PROMPT = """
    Analyze this research query and determine its complexity:
    
    Query: {query}
    Research Plan: {plan}
    
    Classify as:
    - "simple": Single factual answer (e.g., "What is X?")
    - "moderate": Comparison or multi-faceted (e.g., "Compare A and B")
    - "complex": Deep research, multiple dimensions (e.g., "Analyze the impact of X on Y over time")
    
    Return JSON:
    {{
      "complexity": "simple" | "moderate" | "complex",
      "recommended_subagents": number,
      "recommended_tool_calls_per_agent": number,
      "reasoning": "explanation"
    }}
    """
    
    # Use reasoning model
    result = generate_structured_output(COMPLEXITY_ANALYSIS_PROMPT, ...)
    
    return result

def run_deep_research(user_query: str) -> str:
    # Determine scale
    scale = determine_research_complexity(user_query, research_plan)
    
    # Adjust subtask splitting
    MAX_SUBTASKS = {
        "simple": 1,
        "moderate": 4,
        "complex": 12
    }[scale["complexity"]]
    
    # Pass to splitter
    subtasks = split_into_subtasks(
        research_plan,
        max_subtasks=MAX_SUBTASKS,
        depth=scale["recommended_tool_calls_per_agent"]
    )
    
    # Inform coordinator
    coordinator_prompt = COORDINATOR_PROMPT_TEMPLATE.format(
        ...,
        complexity=scale["complexity"],
        expected_depth=scale["reasoning"]
    )
```

---

## 7. EVALUATION FRAMEWORK

### Recommended Approach

**Model for Evaluation:** Use a different, high-quality model than execution models

```python
# For evaluation, use Claude or GPT-4
EVALUATION_MODEL = "anthropic/claude-3.5-sonnet"  # via OpenRouter
# or
EVALUATION_MODEL = "openai/gpt-4o"  # Large context, good at eval
```

### Evaluation Criteria (from Anthropic)

```python
EVALUATION_RUBRIC = """
Evaluate the research report on these dimensions (0.0-1.0 scale):

1. **Factual Accuracy** (0.0-1.0)
   - Do claims match sources?
   - Are there hallucinations?
   
2. **Citation Accuracy** (0.0-1.0)
   - Are sources properly cited?
   - Do citations match claims?
   
3. **Completeness** (0.0-1.0)
   - All aspects of query covered?
   - Appropriate depth?
   
4. **Source Quality** (0.0-1.0)
   - Primary sources used?
   - Authoritative and recent?
   
5. **Tool Efficiency** (0.0-1.0)
   - Right tools used?
   - Reasonable number of calls?
   
Return JSON:
{{
  "factual_accuracy": float,
  "citation_accuracy": float,
  "completeness": float,
  "source_quality": float,
  "tool_efficiency": float,
  "overall_score": float,  // average
  "pass": boolean,  // true if overall > 0.7
  "feedback": "detailed explanation"
}}
"""

def evaluate_research_output(
    user_query: str,
    research_plan: str,
    final_report: str,
    sources: List[dict],
    tool_calls: List[dict]
) -> dict:
    """
    Evaluate the quality of research output.
    """
    
    from openai import OpenAI
    
    eval_client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"]
    )
    
    evaluation = eval_client.chat.completions.create(
        model="anthropic/claude-3.5-sonnet",
        messages=[
            {"role": "system", "content": EVALUATION_RUBRIC},
            {"role": "user", "content": f"""
Query: {user_query}
Plan: {research_plan}
Report: {final_report}
Sources: {json.dumps(sources[:50])}  # Limit context
Tool Calls: {len(tool_calls)} total calls
            """}
        ],
        response_format={"type": "json_object"}
    )
    
    return json.loads(evaluation.choices[0].message.content)
```

### Build an Evaluation Dataset

```python
# Store queries and evaluations
EVAL_DATASET = [
    {
        "query": "Research about climate in Northern France",
        "expected_topics": ["temperature", "precipitation", "seasons", "trends"],
        "expected_sources": ["Météo-France", "academic papers"],
        "human_rating": 0.85,
        "timestamp": "2026-01-31"
    },
    # ... more examples
]

def run_eval_suite():
    """
    Run evaluation on a test set of queries.
    """
    results = []
    
    for eval_case in EVAL_DATASET:
        output = run_deep_research(eval_case["query"])
        evaluation = evaluate_research_output(...)
        
        results.append({
            "query": eval_case["query"],
            "automated_score": evaluation["overall_score"],
            "human_baseline": eval_case["human_rating"],
            "delta": evaluation["overall_score"] - eval_case["human_rating"]
        })
    
    return pd.DataFrame(results)
```

---

## 8. ERROR HANDLING & OBSERVABILITY

### Current Issues
- No explicit error handling
- No logging/tracing of agent decisions
- No retry logic for failed searches

### Recommended Implementation

#### A. Graceful Error Handling

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    reraise=True
)
def call_subagent_with_retry(subtask_id, subtask_title, subtask_description):
    """
    Retry logic for subagent execution.
    """
    try:
        return subagent.run(subagent_prompt)
    except Exception as e:
        logger.error(f"Subagent {subtask_id} failed: {str(e)}")
        
        # Let model know about the error
        error_context = f"Previous attempt failed: {str(e)}. Please try a different approach."
        subagent_prompt_with_context = subagent_prompt + "\n\n" + error_context
        
        raise  # Will trigger retry

def initialize_subagent_safe(subtask_id, subtask_title, subtask_description):
    """
    Safe wrapper with fallback behavior.
    """
    try:
        return call_subagent_with_retry(subtask_id, subtask_title, subtask_description)
    except Exception as e:
        # After all retries failed
        logger.error(f"Subagent {subtask_id} failed after all retries")
        
        # Return partial result
        return f"""
        # {subtask_title} (INCOMPLETE)
        
        ## Error
        Research for this subtask could not be completed: {str(e)}
        
        ## Recommendation
        This section requires manual research or retry.
        """
```

#### B. Observability with LangSmith

```python
from langsmith import Client
from langsmith.run_helpers import traceable

# Initialize LangSmith
langsmith_client = Client(api_key=os.environ["LANGSMITH_API_KEY"])

@traceable(
    run_type="chain",
    name="deep_research_pipeline",
    project_name="deep-research-agent"
)
def run_deep_research(user_query: str) -> str:
    # All function calls within this will be traced
    
    with traceable(name="generate_plan"):
        research_plan = generate_research_plan(user_query)
    
    with traceable(name="split_tasks"):
        subtasks = split_into_subtasks(research_plan)
    
    with traceable(name="coordinator_execution"):
        final_report = coordinator.run(coordinator_prompt)
    
    return final_report
```

#### C. Structured Logging

```python
import logging
import json
from datetime import datetime

# Setup structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('research_agent.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def log_research_event(event_type, data):
    """
    Log structured events for analysis.
    """
    event = {
        "timestamp": datetime.utcnow().isoformat(),
        "event_type": event_type,
        "data": data
    }
    logger.info(json.dumps(event))

# Usage:
log_research_event("research_started", {
    "query": user_query,
    "session_id": session_id
})

log_research_event("subagent_spawned", {
    "subtask_id": subtask_id,
    "subtask_title": subtask_title
})

log_research_event("search_executed", {
    "query": search_query,
    "results_count": len(results),
    "tool": "firecrawl_search"
})
```

#### D. Real-time Progress Tracking (from open-deep-research)

```python
from dataclasses import dataclass
from typing import Literal

@dataclass
class ResearchProgress:
    current_depth: int = 0
    max_depth: int = 7
    completed_steps: int = 0
    total_expected_steps: int = 0
    current_activity: str = ""
    sources_found: int = 0
    
def track_progress(progress: ResearchProgress):
    """
    Update progress in real-time (can be displayed in UI).
    """
    percentage = (progress.completed_steps / progress.total_expected_steps * 100) if progress.total_expected_steps > 0 else 0
    
    print(f"""
    Progress: {percentage:.1f}%
    Depth: {progress.current_depth}/{progress.max_depth}
    Steps: {progress.completed_steps}/{progress.total_expected_steps}
    Activity: {progress.current_activity}
    Sources: {progress.sources_found}
    """)
```

---

## 9. CURRENT SYSTEM STRENGTHS

Your current implementation is already quite good:

✅ **Good Model Selection**
- Qwen3-Next-80B thinking for planning
- MiniMax-M1-80k for large context needs
- GLM-4.7 for subtask generation

✅ **Clean Separation of Concerns**
- Planner → Splitter → Coordinator → Subagents
- Each has a specific role

✅ **Structured Prompts**
- Clear instructions for each component
- Well-defined output formats

✅ **MCP Integration**
- Using Firecrawl for search/scrape
- Good foundation for tool expansion

---

## 10. SUGGESTED IMPROVEMENTS PRIORITY

### HIGH PRIORITY (Implement First)

1. **Add Checkpointing** (LangGraph + PostgreSQL)
   - Enables resumption
   - Saves cost on retries
   - Enables evaluation over time

2. **Better Error Handling**
   - Retry logic
   - Graceful degradation
   - Error context to models

3. **Citation Agent**
   - Separate agent at end
   - Improves credibility
   - Matches Anthropic's approach

### MEDIUM PRIORITY

4. **Improved Task Decomposition**
   - More detailed subtask descriptions
   - Validation step
   - Complexity-based scaling

5. **Extended Thinking**
   - Add to coordinator
   - Add to subtask splitting
   - Add to subagent search strategy

6. **Observability**
   - LangSmith integration
   - Structured logging
   - Progress tracking

### LOW PRIORITY (Nice to Have)

7. **Evaluation Framework**
   - Build eval dataset
   - Automated scoring
   - A/B testing different approaches

8. **Advanced Firecrawl Features**
   - Batch scraping
   - Website crawling
   - Screenshot capture

---

## 11. ALTERNATIVE OPEN-SOURCE MODELS

### Better Than Current Choices

**For Reasoning/Planning:**
```python
# Current: Qwen/Qwen3-Next-80B-A3B-Thinking ✅
# Alternative: DeepSeek-R1 (newer, very good at reasoning)
MODEL = "deepseek-ai/DeepSeek-R1"
# via OpenRouter or HF Inference API
```

**For Large Context Coordination:**
```python
# Current: MiniMax-M1-80k ✅ (Good choice - 80k context)
# Alternative: Qwen/Qwen3-Next-32B (if need more capability)
# Alternative: Mistral/Mixtral-8x22B (better reasoning, smaller context)
```

**For Citation (Structured Output):**
```python
# Suggested: Qwen/Qwen3-Next-80B or Qwen3.5
# These are excellent at following structured output formats
```

**For Evaluation:**
```python
# Use commercial models via OpenRouter (more reliable):
"anthropic/claude-3.5-sonnet"
"openai/gpt-4o"
"google/gemini-pro-1.5"
```

---

## 12. FRAMEWORK RECOMMENDATION: Stay with smolagents for now, BUT...

### Why NOT Migrate Immediately

1. **Your current implementation works**
2. **smolagents + MCP is simpler** for your use case
3. **Less refactoring needed** to add improvements

### When to Consider LangGraph

Migrate to LangGraph if you need:
- ✅ **Built-in checkpointing** (vs custom implementation)
- ✅ **Complex state management** (multiple parallel branches)
- ✅ **Production deployment** (LangSmith observability)
- ✅ **Human-in-the-loop** at specific steps

### Hybrid Approach (RECOMMENDED)

```python
# Keep smolagents for agent execution
# Add LangGraph ONLY for persistence

from smolagents import ToolCallingAgent
from langgraph.checkpoint.postgres import PostgresSaver

# Your existing agents
coordinator = ToolCallingAgent(...)
subagent = ToolCallingAgent(...)

# Add persistence layer
checkpointer = PostgresSaver.from_conn_string(os.environ["POSTGRES_URL"])

# Wrap execution with checkpointing
def run_with_checkpointing(user_query, thread_id):
    config = {"configurable": {"thread_id": thread_id}}
    
    # Check if resuming
    state = checkpointer.get(config)
    if state:
        # Resume from checkpoint
        research_plan = state["research_plan"]
        subtasks = state["subtasks"]
        completed = state["completed_subtasks"]
    else:
        # New research
        research_plan = generate_research_plan(user_query)
        subtasks = split_into_subtasks(research_plan)
        completed = []
    
    # Execute
    for subtask in subtasks:
        if subtask["id"] not in completed:
            try:
                result = initialize_subagent(...)
                completed.append(subtask["id"])
                
                # Save checkpoint
                checkpointer.put(config, {
                    "research_plan": research_plan,
                    "subtasks": subtasks,
                    "completed_subtasks": completed,
                    "results": results
                })
            except Exception as e:
                # Can resume from here later
                logger.error(f"Failed at subtask {subtask['id']}")
                raise
    
    return final_report
```

---

## 13. NEXT STEPS - Recommended Implementation Order

### Phase 1: Foundation (Week 1)
1. Add PostgreSQL checkpointing
2. Implement basic error handling
3. Add structured logging

### Phase 2: Quality (Week 2)
4. Implement citation agent
5. Improve task decomposition prompts
6. Add complexity-based scaling

### Phase 3: Observability (Week 3)
7. Integrate LangSmith
8. Build evaluation framework
9. Add progress tracking UI

### Phase 4: Optimization (Week 4)
10. Add extended thinking to more components
11. Implement advanced Firecrawl features
12. Build evaluation dataset and A/B testing

---

## 14. EXAMPLE: Complete Improved Flow

```python
def run_deep_research_v2(
    user_query: str,
    thread_id: str = None,
    max_retries: int = 3
) -> dict:
    """
    Improved research flow with all enhancements.
    """
    
    # 1. Setup
    thread_id = thread_id or f"research_{datetime.now().timestamp()}"
    config = {"configurable": {"thread_id": thread_id}}
    
    # 2. Checkpointing
    checkpoint = checkpointer.get(config)
    if checkpoint:
        logger.info(f"Resuming research from checkpoint")
        # Resume from saved state
        research_plan = checkpoint["research_plan"]
        subtasks = checkpoint["subtasks"]
        completed_subtasks = checkpoint.get("completed_subtasks", [])
    else:
        # 3. Determine complexity
        complexity = determine_research_complexity(user_query)
        log_research_event("complexity_determined", complexity)
        
        # 4. Generate plan with thinking
        research_plan = generate_research_plan(user_query)
        
        # 5. Split into subtasks
        subtasks = split_into_subtasks(research_plan, max_subtasks=complexity["recommended_subagents"])
        
        # 6. Validate subtasks
        subtasks = validate_subtasks(subtasks, research_plan)
        
        completed_subtasks = []
    
    # 7. Execute with error handling
    results = []
    for subtask in subtasks:
        if subtask["id"] in completed_subtasks:
            continue
            
        try:
            result = call_subagent_with_retry(
                subtask["id"],
                subtask["title"],
                subtask["description"]
            )
            results.append(result)
            completed_subtasks.append(subtask["id"])
            
            # Checkpoint after each subtask
            checkpointer.put(config, {
                "research_plan": research_plan,
                "subtasks": subtasks,
                "completed_subtasks": completed_subtasks,
                "results": results
            })
            
        except Exception as e:
            logger.error(f"Subtask {subtask['id']} failed after retries: {e}")
            # Continue with other subtasks
    
    # 8. Coordinator synthesis
    final_report = coordinator.run(create_coordinator_prompt(results))
    
    # 9. Citation agent
    cited_report = add_citations(final_report, all_sources)
    
    # 10. Evaluation
    evaluation = evaluate_research_output(
        user_query, research_plan, cited_report, all_sources, tool_calls
    )
    
    log_research_event("research_completed", {
        "thread_id": thread_id,
        "evaluation_score": evaluation["overall_score"],
        "subtasks_completed": len(completed_subtasks),
        "total_sources": len(all_sources)
    })
    
    return {
        "report": cited_report,
        "evaluation": evaluation,
        "metadata": {
            "thread_id": thread_id,
            "complexity": complexity,
            "subtasks": len(subtasks),
            "sources": len(all_sources)
        }
    }
```

---

## CONCLUSION

Your current system is a solid foundation. The key improvements are:

1. **Memory/Persistence** - PostgreSQL + LangGraph checkpointing
2. **Citation Agent** - Separate agent at the end for proper attribution
3. **Error Handling** - Retry logic and graceful degradation
4. **Better Prompts** - More detailed task decomposition
5. **Observability** - LangSmith + structured logging
6. **Evaluation** - Automated quality assessment

**Framework Recommendation:** Stay with smolagents + add LangGraph checkpointing as a hybrid approach.

**Priority:** Start with checkpointing, error handling, and citation agent. These provide immediate value with minimal refactoring.

**Timeline:** Can implement all high-priority items in 1-2 weeks.
