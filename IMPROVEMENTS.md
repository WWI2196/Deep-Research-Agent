# Deep Research Agent Improvements

## Based on Anthropic's Multi-Agent Research System

This document outlines the improvements made to the deep research agent based on Anthropic's engineering article about their multi-agent research system.

---

## Key Improvements Implemented

### 1. **"Thinking" Reflection Node & Smart Loop**
- **What Changed**: Replaced simple quality scoring with an intelligent "Thinking" node (LLM) that reviews reports.
- **Why**: A heuristic score is often inaccurate. The LLM can explicitly identify missing information, contradictions, or knowledge gaps.
- **Impact**: System generates specific follow-up subtasks only when gaps exist.
- **Smart Loop**: The system tracks `completed_subtasks` to ensure agents never duplicate work, even across multiple iterations.

**New State Fields**:
- `iteration_count`: Tracks current iteration
- `max_iterations`: Configurable limit (env: `MAX_RESEARCH_ITERATIONS`, default: 2)
- `research_complete`: Boolean flag for completion
- `completed_subtasks`: List of IDs tracking finished work

### 2. **Context Compression (Updated Subagent Prompt)**
- **What Changed**: Subagents now produce "distilled insights" with high information density.
- **Why**: Allows the synthesis agent to process more information without hitting context limits.
- **Impact**: Richer final reports that synthesize findings from more sources without overflow.

### 3. **Deep Firecrawl Integration**
- **What Changed**: Configured Firecrawl as the core engine for search and extraction.
- **Why**: Provides superior markdown extraction and handles dynamic content better than basic scraping.
- **Impact**: Higher success rate in retrieving content from complex websites.
- **Configuration**: Requires `FIRECRAWL_API_KEY`. Includes auto-retries and exponential backoff.

### 4. **Enhanced Task Delegation**
- **What Changed**: Subtasks now include 8 detailed fields instead of 3
- **Why**: Anthropic found that detailed task descriptions prevent agents from duplicating work or misinterpreting tasks
- **Impact**: Clearer division of labor, better results

**New Subtask Fields**:
- `objective`: What the subagent should accomplish
- `output_format`: Expected structure
- `tool_guidance`: Which tools to prioritize
- `source_types`: Preferred sources (academic, official, etc.)
- `boundaries`: What to exclude to avoid overlap

### 5. **Source Quality Scoring & Filtering**
- **What Changed**: Added `_score_source_quality()` function that scores sources 0.0-1.0
- **Why**: Anthropic found early agents chose SEO-optimized content farms over authoritative sources
- **Impact**: Prioritizes primary sources (.gov, .edu, academic journals) over low-quality content

**Quality Indicators**:
- **High Quality** (+0.3): .gov, .edu, arxiv, academic, official, whitepaper
- **Low Quality** (-0.3): listicle, clickbait, "you won't believe", viral

### 6. **Memory System for Context Preservation**
- **What Changed**: Added `memory` field in state to store critical context
- **Why**: Anthropic stores research plans externally when approaching 200k token limit
- **Impact**: Prevents context loss during long research sessions

**Stored in Memory**:
- Research plan
- Subtask definitions
- Can be extended for additional critical context

### 7. **Breadth-First Search Strategy**
- **What Changed**: Updated search query generation to start with SHORT, BROAD queries
- **Why**: Anthropic found agents defaulted to overly specific queries that returned few results
- **Impact**: Better initial exploration before drilling into specifics

### 8. **Graceful Error Handling**
- **What Changed**: Parallel execution now uses `return_exceptions=True` and continues despite individual failures
- **Why**: Anthropic emphasized that minor errors shouldn't derail entire multi-agent system
- **Impact**: System resilience - one failing subagent doesn't crash the entire research

### 9. **Interleaved Thinking Guidance**
- **What Changed**: Enhanced subagent prompts with explicit instructions to evaluate results and adapt
- **Why**: Anthropic uses interleaved thinking after tool results to assess quality and refine next steps
- **Impact**: More intelligent search refinement

**Prompt Instructions Added**:
1. Start with BROAD queries, then narrow
2. Prefer PRIMARY sources over secondary
3. Evaluate after each search: gaps? quality?
4. Adapt strategy based on findings
5. Use interleaved thinking for reflection

### 10. **Source Deduplication & Quality Sorting**
- **What Changed**: Sources are now sorted by quality score and deduplicated by URL
- **Why**: Prevents duplicate work and ensures high-quality sources are prioritized for extraction
- **Impact**: More efficient extraction phase, better evidence quality

### 11. **SOTA Open Source Models Upgrade**
- **What Changed**: Switched to DeepSeek R1, Llama 3.3, and Qwen 2.5
- **Why**: Leverage state-of-the-art open-source models for reasoning and synthesis
- **Impact**: Improved reasoning capabilities and cost-effectiveness
- **Utility**: Added `_clean_deepseek_think` utility to handle DeepSeek's chain-of-thought output tokens

### 12. **Parallel Tool Calling**
- **What Changed**: Subagents now execute tool calls in parallel where possible
- **Why**: Anthropic runs 3+ tools in parallel per subagent to drastically reduce wait times
- **Impact**: Significant reduction in overall research latency (up to 90% faster)

### 13. **Query Expansion (Multi-Query)**
- **What Changed**: `generate_search_queries` now generates 4-7 queries of diverse types: Broad, Specific, Natural Language, and Entity-Centric.
- **Why**: Research by LangChain/LlamaIndex shows that single queries often fail due to vocabulary mismatch.
- **Impact**: Higher recall of relevant documents by covering multiple semantic angles.

### 14. **Source Routing (Dynamic Operator Injection)**
- **What Changed**: The subagent logic intelligently adds domain-specific operators (e.g., `site:.edu`) to *some* queries while preserving broad search capability.
- **Why**: Ensures we capture authoritative sources without missing valuable information from blogs, forums, or unexpected sites.
- **Impact**: Balanced retrieval that values "information is information" while filtering distinct clickbait.

### 15. **Verification & Strict Citation**
- **What Changed**: `CITATION_PROMPT` now acts as a verification step, instructing the model to only cite claims supported by source text and avoiding hallucinations.
- **Why**: Reduces the "fluent hallucination" problem where agents cite plausible but wrong sources.
- **Impact**: Higher reliability of the final report.

---

## Architecture Changes

### Before (Linear Pipeline):
```
init → plan → split → scale → subagents → synthesize → cite → END
```

### After (Iterative with Reflection):
```
init → plan → split → scale → subagents → reflection
                        ↑                       ↓
                        └─────(if gaps found)
                                      ↓
                              (if satisfied)
                                      ↓
                            synthesize → cite → END
```

---

## Configuration Options

Add to your `.env` file:

```bash
# Maximum research iterations (default: 2)
MAX_RESEARCH_ITERATIONS=2

# Firecrawl retry settings
FIRECRAWL_MAX_RETRIES=3
FIRECRAWL_BACKOFF_BASE=0.5
FIRECRAWL_API_KEY=fc-...

# Model Configuration
PLANNER_MODEL_ID=deepseek-ai/DeepSeek-R1-Distill-Llama-70B
TASK_SPLITTER_MODEL_ID=meta-llama/Llama-3.3-70B-Instruct
SCALING_MODEL_ID=deepseek-ai/DeepSeek-R1-Distill-Llama-70B
SUBAGENT_MODEL_ID=Qwen/Qwen2.5-72B-Instruct
COORDINATOR_MODEL_ID=Qwen/Qwen2.5-72B-Instruct
CITATION_MODEL_ID=Qwen/Qwen2.5-72B-Instruct
```

---

## Anthropic's Key Metrics

From their internal evaluations:
- Multi-agent Opus 4 + Sonnet 4 **outperformed single-agent Opus 4 by 90.2%**
- Token usage explains **80% of performance variance**
- Multi-agent systems use **15× more tokens** than chat
- Parallel tool calling reduced research time by **up to 90%**

---

## Still To Implement (Future Enhancements)

Based on Anthropic's article, these would be valuable additions:

### 1. **Extended Thinking Mode**
- Use Claude's native extended thinking for planning
- Requires switching to Anthropic API instead of HuggingFace

### 2. **Tool-Testing Agent**
- Auto-improve tool descriptions by testing and rewriting
- Anthropic achieved 40% faster completion with this

### 3. **Asynchronous Subagent Execution**
- Lead agent could steer subagents mid-execution
- Subagents could coordinate with each other
- More complex state management needed

### 4. **Artifact System**
- Subagents write outputs to filesystem/database
- Lead agent receives lightweight references
- Prevents information loss through "telephone game"

### 5. **Rainbow Deployments**
- Gradual traffic shifting for production updates
- Prevents disrupting running agents

### 6. **Advanced Observability**
- Monitor agent decision patterns
- Track interaction structures
- Production tracing for debugging

---

## Expected Improvements

Based on Anthropic's results, you should see:

1. **Better breadth-first research**: More comprehensive initial exploration
2. **Higher quality sources**: Academic and official sources prioritized
3. **Adaptive research**: System can recognize incomplete research and iterate
4. **Resilience**: Individual failures don't crash the system
5. **Better delegation**: Clearer subtasks reduce duplication
