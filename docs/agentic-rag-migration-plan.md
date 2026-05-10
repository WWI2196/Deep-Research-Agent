# Agentic RAG 架构迁移计划

> 版本：v1.0
> 日期：2026-05-09
> 作者：Claude Code
> 目标：将 Deep Research Agent 的 subagent 从硬编码 6 步流程升级为 Tool-based ReAct Agent，同时保留现有工程优化

---

## 1. 总体目标

将当前 `run_subagent` 的固定流水线：

```
generate_search_queries → parallel_search → batch_evaluate → extract_fulltext → build_evidence → write_report
```

改造为 **ReAct Agent 架构**：

```
SubagentAgent (LLM + Tools)
  ├─ searxng_search(query, limit) → search results
  ├─ document_hybrid_search(query, collection_ids, top_k) → RAG chunks
  ├─ evaluate_sources(candidates, objective) → scored + selected
  ├─ fetch_fulltext(urls) → extracted markdown
  └─ submit_report(evidence, analysis) → final report
```

**关键约束**：规则查询生成、批量评估、证据压缩等优化必须保留为 Tool 内部实现，不能被 LLM 取代。

---

## 2. 架构对比

### 2.1 当前架构（硬编码 Subagent）

```
graph TD
    A[plan_node] --> B[split_node]
    B --> C[subagents_node]
    C --> D[reflection_node]
    D -- gaps --> C
    D -- complete --> E[synthesize_node]
    E --> F[cite_node]

    subgraph "subagent 内部（硬编码）"
        C1[rules queries] --> C2[parallel search]
        C2 --> C3[batch_evaluate_sources]
        C3 --> C4[extract fulltext]
        C4 --> C5[build evidence]
        C5 --> C6[write report]
    end
```

### 2.2 目标架构（Tool-based Agent）

```
graph TD
    A[plan_node] --> B[split_node]
    B --> C[subagents_node]
    C --> D[reflection_node]
    D -- structured gap指令 --> C
    D -- complete --> E[synthesize_node]
    E --> F[cite_node]

    subgraph "subagent 内部（ReAct Agent）"
        S1[LLM决策] -->|调用| T1[searxng_search]
        S1 -->|调用| T2[document_hybrid_search]
        S1 -->|调用| T3[evaluate_and_select]
        S1 -->|调用| T4[fetch_fulltext]
        S1 -->|调用| T5[submit_report]
        T1 --> S1
        T2 --> S1
        T3 --> S1
        T4 --> S1
        T5 --> S1
    end

    subgraph "reflection 协商"
        D -->|gap指令| C
    end
```

---

## 3. 详细改动清单

### Phase 1：基础设施（新增文件）

#### 3.1.1 新增 `src/backend/tools.py`

**职责**：定义所有可被 Agent 调用的工具函数，每个工具都是一个纯函数，接收参数 → 执行 → 返回结构化结果。

**工具清单**：

| 工具名 | 输入 | 输出 | 内部实现 |
|--------|------|------|----------|
| `searxng_search` | `query: str, limit: int = 8` | `{"results": [...]}` | 封装 `search_mod.search` + 空结果回退 (`generate_broader_queries`) + query_cache 读写 |
| `document_hybrid_search` | `query: str, collection_ids: list[str], top_k: int = 12` | `{"results": [...]}` | 封装 `_search_document_collections`，输出标准化为统一格式 |
| `evaluate_sources` | `candidates: list[dict], objective: str` | `{"scored": [...], "selected_for_fulltext": [...]}` | 封装 `batch_evaluate_sources` + 多样性 enforcement + source_type quota |
| `fetch_fulltext` | `urls: list[str]` | `{"extracted": {"url": "text", ...}}` | 封装 `search_mod.extract` (trafilatura) + 异步并发 |
| `submit_report` | `evidence: list[dict], subtask: dict, user_query: str` | `{"report": "markdown"}` | 封装当前的 report generation prompt + 重试逻辑 |

**关键设计决策**：
- 每个工具内部仍保留现有的硬编码优化（如规则查询生成在 `searxng_search` 内部作为 fallback）
- 工具函数签名要简单，方便 LLM 通过 JSON 调用
- 工具返回必须包含 `success: bool` 和 `error: str | None`，方便 Agent 做错误处理

**代码骨架**：

```python
# src/backend/tools.py
from typing import Any, Callable

class Tool:
    def __init__(self, name: str, description: str, params_schema: dict, fn: Callable):
        self.name = name
        self.description = description
        self.params_schema = params_schema
        self.fn = fn

    async def execute(self, **kwargs) -> dict[str, Any]:
        try:
            result = await self.fn(**kwargs)
            return {"success": True, "result": result, "error": None}
        except Exception as e:
            return {"success": False, "result": None, "error": str(e)}

async def searxng_search_tool(query: str, limit: int = 8, query_cache: dict | None = None) -> dict:
    """Search the web using SearXNG. Falls back to broader queries on empty results."""
    ...

async def document_hybrid_search_tool(
    query: str, collection_ids: list[str], top_k: int = 12
) -> dict:
    """Search private document collections using hybrid retrieval (vector + BM25 + RRF)."""
    ...

async def evaluate_sources_tool(
    candidates: list[dict], objective: str, config: Any | None = None
) -> dict:
    """Evaluate source quality and select which ones deserve full-text extraction."""
    ...

async def fetch_fulltext_tool(urls: list[str]) -> dict:
    """Fetch full-text content from URLs via trafilatura."""
    ...

# 工具注册表
RESEARCH_TOOLS: list[Tool] = [
    Tool("searxng_search", "...", {"query": "str", "limit": "int"}, searxng_search_tool),
    Tool("document_hybrid_search", "...", {...}, document_hybrid_search_tool),
    Tool("evaluate_sources", "...", {...}, evaluate_sources_tool),
    Tool("fetch_fulltext", "...", {...}, fetch_fulltext_tool),
    Tool("submit_report", "...", {...}, submit_report_tool),
]
```

#### 3.1.2 新增 `src/backend/react_agent.py`

**职责**：实现一个轻量级的 ReAct Agent 循环，供 `subagent.py` 调用。不依赖 LangGraph 的 `create_react_agent`（避免过度封装），自己实现 `think → act → observe` 循环。

**核心逻辑**：

```python
# src/backend/react_agent.py

async def run_react_agent(
    system_prompt: str,
    user_prompt: str,
    tools: list[Tool],
    max_steps: int = 10,
    chat_fn: Callable = chat,
) -> dict[str, Any]:
    """
    轻量级 ReAct 循环：
    1. LLM 生成思考 + 工具调用（JSON 格式）
    2. 执行工具
    3. 将结果追加到对话历史
    4. 直到 LLM 输出 final_answer 或达到 max_steps
    """
    ...
```

**LLM 输出格式约定**（减少解析失败）：

```json
{
  "thought": "I need to search for academic papers on this topic first, then evaluate results.",
  "action": "searxng_search",
  "action_input": {"query": "transformer architecture attention mechanism", "limit": 8}
}
```

或终止状态：

```json
{
  "thought": "I have gathered sufficient evidence and written the report.",
  "final_answer": "# Subtask Title\n\n## Summary\n..."
}
```

**为什么不用 LangGraph 的 create_react_agent**：
- 我们的 subagent 已经在 LangGraph 的一个节点内部，嵌套 LangGraph 会增加调试复杂度
- 自己实现可以更精细地控制循环终止条件、token 预算、tracing 埋点
- 194 测试的稳定性更容易保持

---

### Phase 2：Subagent 重构（修改文件）

#### 3.2.1 修改 `src/backend/subagent.py`

**当前问题**：`run_subagent` 是 280 行的巨型函数，6 个阶段全部硬编码在一起。

**改造方案**：

1. **保留 `generate_search_queries`** — 作为 `searxng_search_tool` 内部的默认策略
2. **保留 `batch_evaluate_sources`** — 作为 `evaluate_sources_tool` 的核心实现
3. **保留 `run_subagents_parallel`** — 外部调用接口不变，内部改为调用新的 ReAct Agent
4. **重写 `run_subagent`** — 从硬编码流程改为 ReAct Agent 驱动

**新的 `run_subagent` 流程**：

```python
async def run_subagent(
    user_query: str,
    research_plan: str,
    subtask: dict[str, Any],
    tool_budget: int,
    query_cache: dict[str, list[dict]] | None = None,
    document_collections: list[str] | None = None,
    gap_instruction: dict[str, Any] | None = None,  # 新增：来自 reflection 的指令
) -> dict[str, Any]:
    """
    新版 run_subagent：ReAct Agent 驱动。
    """
    # 1. 构建工具上下文（注入 query_cache、document_collections）
    tools = _build_tool_instances(query_cache, document_collections)

    # 2. 构建 system prompt（包含 subtask 元数据）
    system_prompt = _build_subagent_system_prompt(subtask, research_plan, user_query)

    # 3. 构建 user prompt（包含 gap_instruction，如果有）
    user_prompt = _build_subagent_user_prompt(subtask, gap_instruction)

    # 4. 运行 ReAct Agent
    result = await run_react_agent(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        tools=tools,
        max_steps=min(tool_budget, 12),
    )

    # 5. 提取 report 和 sources
    report = result.get("final_answer", "")
    tool_calls = result.get("tool_calls", [])  # 用于溯源

    # 6. 后处理（保留现有逻辑）：空报告 fallback、来源提取等
    if len(report) < 200:
        report = _build_empty_report(subtask, tool_calls)

    sources = _extract_sources_from_tool_calls(tool_calls)

    return {
        "subtask_id": subtask["id"],
        "subtask_title": subtask["title"],
        "report": report,
        "sources": sources,
        "evidence_count": len([t for t in tool_calls if t["tool"] in ("searxng_search", "document_hybrid_search")]),
        "tool_calls": tool_calls,  # 新增：详细的工具调用链
    }
```

**向后兼容**：`run_subagents_parallel` 的函数签名完全不变，调用方（`graph.py`）无需修改。

#### 3.2.2 修改 `src/backend/prompts.py`

**新增 `SUBAGENT_REACT_SYSTEM`**：

替换现有的 `SUBAGENT_REPORT`，改为指导 LLM 如何使用工具的 ReAct 风格 prompt：

```
You are a specialized research sub-agent with access to research tools.

Your task: {subtask_title} ({subtask_id})
Objective: {subtask_objective}
Source types to prefer: {subtask_source_types}
Boundaries: {subtask_boundaries}

You have access to these tools:
- searxng_search: Web search via SearXNG
- document_hybrid_search: Search private document collections
- evaluate_sources: Score candidate sources and decide which to deep-read
- fetch_fulltext: Extract full article text from URLs
- submit_report: Submit your final report (call this when done)

Workflow guidance:
1. Start by searching (web + documents in parallel if collections available)
2. Evaluate results to find high-quality sources
3. Fetch full-text for the most promising sources
4. Analyze evidence and write report with inline [src: url] citations
5. Call submit_report with your final markdown

Rules:
- After EVERY factual claim, include [src: <url>] immediately
- Do NOT add a "Sources" or "References" section at the end
- Use submit_report ONLY when you are truly done
```

**保留原有 prompts**：`SOURCE_EVALUATE`、`PLANNER`、`SPLITTER`、`REFLECTION`、`SYNTHESIS`、`FAILURE_SUMMARY` 全部保留，只是 `SUBAGENT_REPORT` 降级为 `submit_report_tool` 内部使用的 prompt。

---

### Phase 3：Reflection → Subagent 协商通道（修改文件）

#### 3.3.1 修改 `src/backend/models.py`

**State 新增字段**：

```python
class ResearchState(TypedDict, total=False):
    # ... 现有字段 ...

    # 新增：Agentic RAG 协商字段
    gap_instructions: list[dict[str, Any]]  # reflection 生成的结构化指令队列
    tool_call_history: list[dict[str, Any]]  # 所有 subagent 的工具调用记录
```

**新增 Pydantic 模型**：

```python
class GapInstruction(BaseModel):
    """Reflection Agent 向 Subagent 发出的补充检索指令。"""
    target_subtask_id: str  # 哪个子任务需要补充
    gap_type: str  # "missing_evidence" | "insufficient_depth" | "contradiction"
    description: str  # 自然语言描述缺口
    suggested_queries: list[str]  # 建议的检索查询
    required_source_types: str | None = None  # 特定来源类型要求
```

#### 3.3.2 修改 `src/backend/graph.py`

**改动点 1：Reflection 输出结构化 gap 指令**

当前 `_reflection_node` 的 gap 是直接在 `s["subtasks"]` 中追加新的 subtask。新架构下，reflection 可以选择两种方式：
- **方式 A**（保持现有）：生成新的 subtask，走完整 subagent 流程
- **方式 B**（新增）：生成 `GapInstruction`，传给现有的 subtask 让其补充检索

在 `_reflection_node` 中增加逻辑：

```python
# 在 reflection 节点中
if new_subtasks:
    # 判断是全新子任务还是现有子任务的补充
    for gap in raw_gaps:
        st = gap.get("subtask", {})
        gap_type = gap.get("gap_type", "new_subtask")

        if gap_type == "supplement_existing" and st.get("target_subtask_id"):
            # 方式 B：生成 GapInstruction，加入队列
            s.setdefault("gap_instructions", []).append({
                "target_subtask_id": st["target_subtask_id"],
                "gap_type": gap.get("gap_detail", "missing_evidence"),
                "description": gap.get("gap_detail", ""),
                "suggested_queries": st.get("keywords", []),
            })
        else:
            # 方式 A：全新子任务
            s["subtasks"].append(st)
```

**改动点 2：Subagent 节点读取 gap 指令**

在 `_subagents_node` 中：

```python
# 为每个待运行的 subtask 查找是否有 gap_instruction
gap_instructions = s.get("gap_instructions", [])
for t in to_run:
    gap = next(
        (g for g in gap_instructions if g["target_subtask_id"] == t["id"]),
        None
    )
    # 将 gap 传给 run_subagent
    tasks.append(run_subagent(
        ..., gap_instruction=gap
    ))
```

**改动点 3：状态清理**

subagent 运行后，已消费的 gap_instruction 从队列中移除：

```python
consumed_ids = {r["subtask_id"] for r in results.get("raw", [])}
s["gap_instructions"] = [
    g for g in s.get("gap_instructions", [])
    if g["target_subtask_id"] not in consumed_ids
]
```

#### 3.3.3 修改 `src/backend/prompts.py` — `REFLECTION`

在 `REFLECTION` prompt 中增加对 `gap_type` 的指导：

```
3. 对于发现的缺口，判断类型：
   - "new_subtask": 需要全新调研的独立主题（生成完整 subtask）
   - "supplement_existing": 现有子任务证据不足，需要补充检索（生成 GapInstruction）

4. 如果缺口可以通过在现有子任务中补充搜索解决（如缺少某类来源、某时间段数据），
   优先使用 supplement_existing，并指定 target_subtask_id。
```

---

### Phase 4：Planner 预检索（可选增强）

#### 3.4.1 修改 `src/backend/planning.py` — `generate_research_plan`

**新增可选预检索**：

```python
async def generate_research_plan(
    user_query: str,
    document_collections: list[str] | None = None,
) -> dict[str, Any]:
    # ... 现有逻辑 ...

    # 新增：如果用户指定了文档库，先做轻量预检索
    if document_collections:
        preview_results = await document_hybrid_search_tool(
            query=user_query,
            collection_ids=document_collections,
            top_k=5,
        )
        if preview_results.get("results"):
            # 将预览结果注入 prompt，帮助 planner 了解文档库中有哪些内容
            preview_text = _format_preview(preview_results["results"])
            # 追加到 planner 的 user message 中
```

**对 planner prompt 的增强**：

```
Available document collections preview:
{preview_text}

If the preview shows relevant documents, tailor your keywords and source_types
to leverage these private sources. Mark dimensions that can be primarily
answered from document library as "source_types": "document".
```

---

### Phase 5：Tracing 与 Observability（修改文件）

#### 3.5.1 修改 `src/backend/tracing.py`

**新增 trace 类型**：

```python
# 在 react_agent.py 中每次 tool call 前后埋点
await trace("subagents", "tool_call_start", f"Calling {tool_name}", {
    "subtask_id": sid,
    "tool": tool_name,
    "input": tool_input,
})

await trace("subagents", "tool_call_end", f"{tool_name} complete", {
    "subtask_id": sid,
    "tool": tool_name,
    "success": result["success"],
    "result_preview": str(result.get("result", ""))[:200],
})
```

#### 3.5.2 前端 `log-viewer.js` 适配

在 timeline 中渲染 tool_call_start/tool_call_end 事件，展示 Agent 的思考-行动-观察循环。

---

## 4. 文件改动总览

| 文件 | 动作 | 改动量 | 说明 |
|------|------|--------|------|
| `src/backend/tools.py` | **新增** | ~300 行 | 5 个工具的实现 + Tool 基类 |
| `src/backend/react_agent.py` | **新增** | ~200 行 | 轻量级 ReAct 循环 |
| `src/backend/subagent.py` | **大幅修改** | ~+50/-200 行 | run_subagent 重写，保留优化逻辑 |
| `src/backend/prompts.py` | **新增+保留** | ~+80 行 | 新增 SUBAGENT_REACT_SYSTEM，保留其余 |
| `src/backend/models.py` | **新增字段** | ~+15 行 | gap_instructions、tool_call_history |
| `src/backend/graph.py` | **修改** | ~+40 行 | reflection 输出 gap 指令、subagent 消费指令 |
| `src/backend/planning.py` | **可选修改** | ~+30 行 | planner 预检索（可延后） |
| `src/backend/tracing.py` | **无修改** | — | 现有 trace() 足够，调用方埋点 |
| `tests/test_agents.py` | **大幅修改** | ~+100 行 | 新增 ReAct Agent 测试、工具测试 |

---

## 5. 数据流变化

### 5.1 当前数据流

```
User Query
  → Planner → Plan (dimensions)
  → Splitter → Subtasks
  → Subagent (硬编码 6 步) → Report + Sources
  → Reflection → [gaps?] → 新 Subtasks
  → Synthesis → Report
  → Citation → Final Report
```

### 5.2 目标数据流

```
User Query
  → Planner → Plan (dimensions) [可选: 预检索文档库]
  → Splitter → Subtasks
  → Subagent (ReAct Agent + Tools)
      ├─ searxng_search / document_hybrid_search
      ├─ evaluate_sources
      ├─ fetch_fulltext
      └─ submit_report
    → Report + Sources + Tool Call History
  → Reflection
      ├─ [全新子任务] → 追加 Subtasks
      └─ [现有子任务补充] → 生成 GapInstruction
  → Subagent (接收 GapInstruction → 补充检索)
  → Synthesis → Report
  → Citation → Final Report
```

---

## 6. 测试策略

### 6.1 新增测试

在 `tests/` 下新增 `test_react_agent.py` 和 `test_tools.py`：

```python
# test_tools.py
async def test_searxng_search_tool_uses_query_cache():
    """验证 query_cache 命中时不会重复搜索。"""

async def test_searxng_search_tool_empty_result_rollback():
    """验证空结果时自动触发 broader queries。"""

async def test_evaluate_sources_tool_preserves_diversity():
    """验证 enforce_source_diversity 仍在工具内部生效。"""

async def test_document_hybrid_search_tool_output_format():
    """验证输出与 searxng_search_tool 格式一致。"""

# test_react_agent.py
async def test_react_agent_runs_tool_and_returns_result():
    """验证基本 think → act → observe 循环。"""

async def test_react_agent_respects_max_steps():
    """验证 max_steps 限制。"""

async def test_react_agent_handles_tool_error():
    """验证工具失败时 Agent 能继续或优雅终止。"""

async def test_react_agent_produces_final_answer():
    """验证最终输出包含 report。"""
```

### 6.2 回归测试

- `tests/test_agents.py` 中的现有 subagent 测试需要更新 mock 方式
- 端到端测试（如果有）应不受影响，因为 `run_subagents_parallel` 签名不变
- 保持 **194 测试全部通过**

### 6.3 集成测试

新增一个集成测试：验证 reflection 生成的 `GapInstruction` 能被 subagent 正确消费并触发补充检索。

---

## 7. 回滚方案

### 7.1 代码层面

- `subagent.py` 保留旧版 `run_subagent` 为 `_run_subagent_legacy`，通过环境变量 `AGENTIC_RAG_ENABLED=0` 切换
- `graph.py` 中通过 flag 决定是否传递 `gap_instruction`

### 7.2 数据层面

- `ResearchState` 新增字段均为 `total=False`，旧数据不会报错
- `gap_instructions` 为空列表时行为与当前完全一致

### 7.3 快速回滚

如果上线后发现问题，只需修改 `graph.py` 中 `_subagents_node` 的调用，不传 `gap_instruction`，并恢复旧 prompt。

---

## 8. 实施顺序与里程碑

| 阶段 | 内容 | 预估工时 | 产出 |
|------|------|----------|------|
| **P0** | 提取 `tools.py`，将现有 subagent 内 6 步流程拆分为独立工具函数 | 1-2 天 | 5 个工具 + 测试 |
| **P1** | 实现 `react_agent.py` 核心循环 | 1 天 | ReAct Agent + 测试 |
| **P2** | 重写 `subagent.py` 的 `run_subagent`，接入 ReAct Agent | 1-2 天 | 新 subagent，194 测试通过 |
| **P3** | 实现 Reflection → Subagent 协商（GapInstruction） | 1 天 | graph.py + models.py 更新 |
| **P4** | Planner 预检索（可选） | 0.5 天 | planning.py 增强 |
| **P5** | 前端 log-viewer 展示 tool call 链 | 0.5-1 天 | UI 增强 |
| **P6** | 端到端验证 + 性能基准测试 | 1-2 天 | 对比报告 |

**总预估**：5-8 天（不含 P5 UI）

---

## 9. 风险与对策

| 风险 | 可能性 | 影响 | 对策 |
|------|--------|------|------|
| LLM 不遵循 tool call 格式 | 中 | 高 | ① 严格的 JSON schema prompt ② 2 次重试 ③ fallback 到旧版硬编码 |
| Token 消耗上升 | 高 | 中 | ① max_steps 限制 ② 轻量级 thought 提示 ③ evaluate_sources 保持批量 |
| 延迟增加 | 高 | 低 | ① 并行 tool call（LLM 一次决定多个搜索）② searxng 本身已并行 |
| 测试覆盖率下降 | 低 | 高 | ① 每个工具独立测试 ② mock LLM 响应固定 tool call 序列 |
| 证据质量下降 | 中 | 高 | ① 保留 batch_evaluate_sources ② submit_report 前强制检查 citations |

---

## 10. 核心设计原则（再次强调）

1. **LLM 决定「做什么」，代码决定「怎么做」**
   - LLM 决定：搜什么主题、从哪搜、要不要继续搜
   - 代码决定：怎么生成查询串、怎么批量评分、怎么压缩证据

2. **渐进式改造，随时可回滚**
   - 每个阶段都保持向后兼容
   - 旧版代码保留为 fallback

3. **测试先行**
   - 新工具必须先有测试再接入 Agent
   - 194 测试是红线

4. **Tracing 全覆盖**
   - 每个 tool call 都要有 trace，方便观测 Agent 决策过程

---

*本计划基于当前代码库状态（commit c1c18b8）制定。如有架构调整需求，请先更新本计划再进入实施。*
