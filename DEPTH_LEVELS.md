# 深度分级机制设计

## 设计原则

- **深度3（质量优先）**：当前~30分钟为锚点，适当加深质量
- **深度2（均衡）**：当前默认行为，控制不重要步骤
- **深度1（快速认知）**：大幅简化，目标5-10分钟

## 三档参数对比

### 1. Planning 阶段

| 参数 | 深度1 | 深度2 | 深度3 |
|------|-------|-------|-------|
| planner模式 | 单次LLM调用 | ReAct Agent | ReAct Agent |
| planner搜索轮次 | 0（跳过搜索） | 3-4次 | 5-6次 |
| planner最大步数 | 1（无工具） | 10步 | 12步 |
| splitter | 跳过，直接映射 | 独立LLM调用 | 独立LLM调用 |
| 子代理数量 | 2-3个 | 4-6个 | 6-8个 |

### 2. Subagent 阶段

| 参数 | 深度1 | 深度2 | 深度3 |
|------|-------|-------|-------|
| 搜索预算/子代理 | 4-6次 | 8-10次 | 12-15次 |
| ReAct步数上限 | 8-10步 | 15步 | 18-20步 |
| 评估 | 跳过，用搜索引擎分数 | 批量评估（batch=20） | 批量评估+更严格阈值 |
| 全文提取数量 | top-2 | top-5 | top-8 |
| 搜索轮次上限 | 2轮 | 3-4轮 | 5-6轮 |
| 空结果回退 | 跳过 | 保留 | 保留 |

### 3. Reflection 阶段

| 参数 | 深度1 | 深度2 | 深度3 |
|------|-------|-------|-------|
| max_iterations | 1 | 2 | 3 |
| 质量阈值 | 0.5 | 0.65 | 0.75 |
| gap数量上限 | 0（跳过gap） | 3 | 5 |
| 最小改进门控 | N/A | 0.08 | 0.05 |

### 4. Synthesis 阶段

| 参数 | 深度1 | 深度2 | 深度3 |
|------|-------|-------|-------|
| 综合输入上限 | 40000字符 | 80000字符 | 120000字符 |
| 截断恢复轮次 | 2轮 | 4轮 | 6轮 |
| 薄弱章节深化 | 跳过 | 最多3个（阈值800字符） | 最多8个（阈值600字符） |
| task compliance | 跳过 | 保留 | 保留 |
| 深化并发度 | N/A | 3 | 5 |

### 5. 持久化与追踪

| 参数 | 深度1 | 深度2 | 深度3 |
|------|-------|-------|-------|
| checkpoint频率 | 仅init/subagents/cite | 全部节点 | 全部节点 |
| trace级别 | warning+ | info+ | debug+ |

## 预估时间

| 深度 | 预估时间 | LLM调用次数 | 来源数量 |
|------|----------|-------------|----------|
| 深度1 | 5-10分钟 | 20-30次 | 20-40个 |
| 深度2 | 20-25分钟 | 60-80次 | 80-120个 |
| 深度3 | 30-40分钟 | 100-130次 | 120-180个 |

## 实现方案

### 核心设计：DepthProfile 数据类

```python
@dataclass
class DepthProfile:
    # Planning
    planner_use_react: bool = True
    planner_max_steps: int = 10
    planner_search_rounds: int = 4
    use_splitter: bool = True
    max_subagents: int = 6
    
    # Subagent
    search_budget_per_subagent: int = 10
    react_max_steps: int = 15
    evaluate_sources: bool = True
    evaluate_batch_size: int = 20
    fulltext_top_k: int = 5
    max_search_rounds: int = 4
    empty_result_rollback: bool = True
    
    # Reflection
    max_iterations: int = 2
    quality_threshold: float = 0.65
    max_gaps: int = 3
    min_improvement_gate: float = 0.08
    
    # Synthesis
    max_input_chars: int = 80000
    continuation_max_rounds: int = 4
    deepen_thin_sections: bool = True
    deepen_max_sections: int = 5
    deepen_char_threshold: int = 800
    deepen_citation_threshold: int = 3
    deepen_concurrency: int = 3
    verify_compliance: bool = True
    
    # Persistence
    checkpoint_frequency: str = "all"  # "minimal" | "all"
    trace_level: str = "info"  # "warning" | "info" | "debug"
```

### 三档预设

```python
DEPTH_PROFILES = {
    1: DepthProfile(
        planner_use_react=False,
        planner_max_steps=1,
        planner_search_rounds=0,
        use_splitter=False,
        max_subagents=3,
        search_budget_per_subagent=6,
        react_max_steps=10,
        evaluate_sources=False,
        fulltext_top_k=2,
        max_search_rounds=2,
        empty_result_rollback=False,
        max_iterations=1,
        quality_threshold=0.5,
        max_gaps=0,
        max_input_chars=40000,
        continuation_max_rounds=2,
        deepen_thin_sections=False,
        verify_compliance=False,
        checkpoint_frequency="minimal",
        trace_level="warning",
    ),
    2: DepthProfile(
        # 所有参数使用默认值（当前行为）
    ),
    3: DepthProfile(
        planner_search_rounds=6,
        planner_max_steps=12,
        max_subagents=8,
        search_budget_per_subagent=15,
        react_max_steps=20,
        fulltext_top_k=8,
        max_search_rounds=6,
        max_iterations=3,
        quality_threshold=0.75,
        max_gaps=5,
        min_improvement_gate=0.05,
        max_input_chars=120000,
        continuation_max_rounds=6,
        deepen_max_sections=8,
        deepen_char_threshold=600,
        deepen_concurrency=5,
        trace_level="debug",
    ),
}
```

## 实现步骤

1. **models.py**：添加 `DepthProfile` 数据类和 `DEPTH_PROFILES` 预设
2. **config.py**：添加深度配置加载逻辑
3. **graph.py**：在 `_init_node` 中根据深度设置 state 参数
4. **planning.py**：`generate_research_plan` 接受深度参数，深度1跳过 ReAct
5. **subagent.py**：`run_subagent` 接受深度参数，控制搜索预算和步数
6. **react_agent.py**：`run_react_agent` 接受深度参数，控制评估和回退
7. **synthesis.py**：`synthesize_report` 接受深度参数，控制深化和验证
8. **前端**：更新深度选择的描述文本
