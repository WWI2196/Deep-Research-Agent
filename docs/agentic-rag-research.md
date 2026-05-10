# Agentic RAG 调研报告

> 调研时间：2026-05-09
> 背景：为 Deep Research Agent 评估从现有 Hybrid RAG（Chroma + bm25s + RRF）向 Agentic RAG 演进的可行性与收益。

---

## 1. 什么是 Agentic RAG

### 1.1 定义

**Agentic RAG**（智能体检索增强生成）是将 LLM Agent 的自主决策能力嵌入传统 RAG 流程的演进架构。传统 RAG 是静态管道（检索 → 填充上下文 → 生成），而 Agentic RAG 让 LLM 在检索环节拥有**动态控制权**——它可以决定是否需要检索、检索什么、如何评估结果、以及是否迭代优化。

核心特征（来自 [Agentic RAG Survey, 2025](https://arxiv.org/html/2501.09136v1)）：

| 维度 | 传统 RAG | Agentic RAG |
|------|----------|-------------|
| 检索策略 | 固定（单次 top-k） | 动态（可选路由、迭代、多源） |
| 查询处理 | 原查询直接嵌入 | 可分解、重写、扩展 |
| 结果评估 | 无（直接塞给 LLM） | 自评分、相关性判断、去噪 |
| 错误修正 | 无 | 循环修正、补充检索、回退策略 |
| 工具使用 | 无 | 可调用搜索引擎、数据库、计算工具等 |

### 1.2 核心架构模式

当前主流的 Agentic RAG 模式可分为四大类：

#### ① Corrective RAG (CRAG)

**原理**：在检索后加入一个「评分+修正」循环。LLM 对检索到的文档进行相关性评分，若评分低于阈值，则触发外部搜索（如 Web Search）补充信息，而非直接使用低质量上下文。

**流程**：`检索 → 相关性评分 → [低质量?] → Web 搜索补充 → 生成`

**特点**：最简单的 Agentic 改造，只需要增加一个判断节点和一个回退工具。

#### ② Self-RAG

**原理**：不仅评估检索文档，还评估生成内容。LLM 在生成过程中输出结构化反思标记（如 `[Retrieve]`、`[No Retrieve]`、`[Support]`、`[Irrelevant]`），根据内容需求动态决定是否需要额外检索。

**流程**：`生成 → 反思标记 → [需要证据?] → 检索 → 重新生成`

**特点**：更细粒度的控制，但要求模型支持特定输出格式或微调。

#### ③ Adaptive RAG

**原理**：根据查询复杂度动态选择检索策略。简单查询直接回答（免检索），中等查询用单次 RAG，复杂查询用迭代/多跳检索。通常用一个轻量级分类器（如 T5 或 LLM 自身）做路由决策。

**流程**：`查询分类 → [简单] 直接生成 / [中等] 标准 RAG / [复杂] 迭代 RAG`

**特点**：性价比最优，避免对简单问题过度检索。

#### ④ Multi-Agent RAG

**原理**：将 RAG 流程拆分为多个专门化 Agent，通过协作完成复杂任务。典型分工：
- **检索 Agent**：负责查询重写和文档检索
- **评估 Agent**：评估检索结果质量
- **生成 Agent**：基于证据生成答案
- **验证 Agent**：验证答案一致性和完整性

**特点**：适合复杂多跳问答和研究型任务，与我们当前 Deep Research Agent 的「Planner → Subagent → Synthesis」架构理念高度契合。

---

## 2. 开源实现参考

### 2.1 官方与高质量实现

| 项目 | 技术栈 | 特点 | 链接 |
|------|--------|------|------|
| **LangGraph Agentic RAG (Official)** | LangGraph, Python | 最权威的参考实现，展示如何用 `create_react_agent` 给 LLM 绑定检索工具 | [GitHub](https://github.com/langchain-ai/langgraph/blob/main/examples/rag/langgraph_agentic_rag.ipynb) |
| **LangChain Docs Tutorial** | LangGraph | 官方教程，涵盖从基础工具绑定到自定义节点边 | [Docs](https://docs.langchain.com/oss/python/langgraph/agentic-rag) |
| **Agentic RAG for Dummies** | LangGraph, 模块化 | 一键切换 LLM 提供商（Ollama/Anthropic/OpenAI），组件完全可替换 | [GitHub](https://github.com/GiovanniPasq/agentic-rag-for-dummies) |
| **MultiAgenticRAG** | LangGraph, Multi-Agent | 多 Agent 协作的 RAG 系统，适合研究型复杂查询 | [GitHub](https://github.com/nicoladisabato/MultiAgenticRAG) |
| **AgenticRAG-Survey** | 综述 | 整理了 Agentic RAG 的论文、实现和评估基准 | [GitHub](https://github.com/asinghcsu/AgenticRAG-Survey) |

### 2.2 当前项目的适配思路

我们的 Deep Research Agent 已经具备 Agentic RAG 的雏形：

```
现有架构（已具备 Agentic 特征）：
  input → planner → split_into_subtasks → run_subagent(检索+评估+生成)
        → reflection → [需要补充?] → 循环 → synthesize → cite
```

当前子代理（`run_subagent`）内部流程已经是 Agentic 的：
1. 规则生成查询（query reformulation）
2. 并行搜索（SearXNG + 文档库 hybrid retrieval）
3. 批量评估（batch_evaluate_sources — Self-RAG 风格的评分）
4. 内容提取（trafilatura）
5. 证据构建与报告生成

**距离完整 Agentic RAG 的差距**：
- 缺少**显式的检索结果质量判断节点**（CRAG 风格：低质量时自动回退/补充）
- 缺少**生成后的自验证循环**（Self-RAG 风格：生成内容是否充分引用证据）
- 缺少**查询复杂度路由**（Adaptive 风格：简单问题直接回答，复杂问题才启动子代理）
- 缺少**多跳检索能力**（当前子代理是单轮的，无迭代深化机制）

---

## 3. 效果对比：Agentic RAG vs 向量检索+BM25

### 3.1 基准测试数据

#### HotpotQA / MultiHop QA（2025 SOTA）

| 方法 | HotpotQA (EM/F1) | 2WikiMultiHop (EM/F1) | MuSiQue (EM/F1) |
|------|------------------|----------------------|-----------------|
| Full Context（无检索） | 44.2 / 58.3 | 43.2 / 52.1 | 19.8 / 29.4 |
| IRCoT（传统迭代检索） | 49.3 / 60.7 | 57.7 / 68.0 | 26.5 / 36.5 |
| **PRISM（Agentic 检索）** | **54.2 / 67.0** | **48.6 / 57.0** | **31.2 / 41.8** |
| Oracle（金标准段落） | 64.8 / 77.8 | 61.4 / 71.1 | 38.8 / 50.9 |

来源：[PRISM: Agentic Retrieval with LLMs for Multi-Hop Question Answering](https://arxiv.org/abs/2510.14278)

**关键发现**：PRISM 在 HotpotQA 上比传统迭代检索 IRCoT 提升 **+4.9 EM / +6.3 F1**，在多跳场景中检索召回率从 61.5% 提升到 **90.9%**。

#### 检索质量与效率（KDD '2025）

某生产级 Agentic Hybrid 系统对比：

| 方案 | 检索准确率 | 延迟 | 相对提升 |
|------|-----------|------|----------|
| BM25 Only | 58% | 8ms |  baseline |
| Hybrid (BM25 + Dense, 无重排) | 79% | 25ms | +21% |
| Cascade + Rerank | 91% | 75ms | +33% |
| **Agentic Routing + Hybrid** | **94.5%** | **43→11s*** | **+36.5%** |

\* 多 Agent 并行后延迟反而降低（从串行 43s 到并行 11s）。

来源：[From BM25 to Agentic RAG: The Evolution of Machine Memory](https://interestingengineering.substack.com/p/from-bm25-to-agentic-rag-the-evolution)

#### 金融/精确数值场景（T2-RAGBench, 2026）

在 23,088 条金融查询上的对比：

| 策略 | Recall@5 | 适用场景 |
|------|---------|----------|
| Dense Retrieval | 中等 | 语义理解型查询 |
| BM25 | **高于 Dense** | 精确数值/术语查询 |
| Hybrid + Neural Rerank | **0.816** | 综合最优 |
| CRAG | 有提升但有限 | 不能超越优质 Hybrid |
| HyDE / Multi-query | 提升有限 | 精确数值场景表现差 |

来源：[From BM25 to Corrective RAG: Benchmarking Retrieval Strategies](https://arxiv.org/html/2604.01733v1)

### 3.2 效果总结

| 对比维度 | Hybrid (Vector+BM25+RRF) | Agentic RAG | 提升幅度 |
|----------|-------------------------|-------------|---------|
| **简单单跳 QA** | 良好 | 良好（Adaptive 免检索更快） | 速度↑，准确率持平 |
| **多跳复杂推理** | 较差（召回率 60-70%） | 优秀（召回率 85-91%） | **+20~30% 召回** |
| **幻觉率** | 中 | 低（自验证机制） | 显著降低 |
| **长尾/边缘查询** | 差（检索为空或低质） | 优（自动回退/重写） | **+15~25% 准确率** |
| **延迟（单次）** | 低（<100ms） | 较高（迭代循环） | 1.5~3x |
| **延迟（端到端复杂任务）** | 高（人工拆解） | 可控（并行 Agent） | 可能更低 |
| **Token 成本** | 低 | 中（额外推理步骤） | 1.2~2x |

### 3.3 重要反例与边界

并非所有场景 Agentic RAG 都更优：

1. **科学文献检索**：传统 BM25 比 LLM-based Agentic 检索高出约 **30%**（因为 Agent 生成的子查询过于偏向关键词，丢失专业术语变体）。解决方案是 Corpus-level 元数据增强（用 LLM 为文档预生成关键词标签）。
2. **精确数值约束**：如果底层检索器（如向量库）本身无法处理数值范围过滤，仅靠 Agentic 编排无法弥补，必须配合结构化元数据过滤。
3. **成本敏感场景**：Self-RAG 的反射步骤会额外消耗 token，在简单查询密集型场景可能得不偿失。

---

## 4. 对当前项目的建议

### 4.1 演进路径（渐进式改造）

基于 Deep Research Agent 现有的 7 节点 LangGraph 架构，建议按以下优先级逐步引入 Agentic 能力：

**Phase 1: CRAG 式检索质量回退（改动最小，收益明确）**

在 `subagent.py` 的 `batch_evaluate_sources` 后增加一个判断节点：
- 若平均质量分 < 0.5 且高质量源 < 3 个 → 触发补充检索（`generate_broader_queries` + 二次搜索）
- 当前已有空结果回退（`empty-result rollback`），可扩展为质量驱动的主动回退

**Phase 2: Adaptive 查询路由（提升性价比）**

在 `plan` 节点后增加复杂度分类器：
- **简单事实查询**（如"Python 是谁发明的"）→ 跳过 subagent，直接单次检索 + 生成
- **分析型查询** → 走完整 subagent 流程
- **研究型查询** → 启用多轮 reflection + 迭代检索

**Phase 3: 多跳迭代检索（冲击多步推理天花板）**

在 `reflection` 节点中增加「证据缺口检测」：
- LLM 评估当前已收集证据是否完整回答子任务
- 若发现缺口（如缺少对比数据、时间线断裂）→ 生成针对性补充查询 → 回到 subagent
- 与我们现有的 `needs_continuation` 机制结合

**Phase 4: 多 Agent 协作（长期目标）**

将检索、评估、生成分解为独立 Agent：
- RetrieverAgent：专精查询改写和检索策略选择
- EvaluatorAgent：专精源质量评估和去噪
- SynthesizerAgent：专精报告生成和一致性检查

### 4.2 预期收益评估

| 改造阶段 | 开发成本 | 预期准确率提升 | 适用场景 |
|---------|---------|---------------|---------|
| Phase 1 (CRAG 回退) | 低（1-2 天） | +5~10% | 检索结果质量波动大的查询 |
| Phase 2 (Adaptive 路由) | 中（3-5 天） | 速度↑30%，成本↓20% | 混合查询负载 |
| Phase 3 (多跳迭代) | 中高（1-2 周） | +10~15%（多跳场景） | 复杂研究型查询 |
| Phase 4 (Multi-Agent) | 高（2-4 周） | +15~20% | 企业级深度研究 |

### 4.3 风险与注意事项

1. **延迟敏感**：Agentic 循环会增加单请求延迟，但 Deep Research Agent 本身就是异步长任务（SSE 流式输出），用户预期是可接受的。
2. **Token 成本**：每次反射/评估都消耗额外 token，建议对轻量判断使用低成本模型（如 Gemini-2.5-Flash-Lite 或本地模型）。
3. **测试覆盖**：引入循环后需特别注意终止条件（max iteration），避免无限循环。现有 `iteration_count` 上限机制可复用。
4. **不要放弃 Hybrid 检索**：2025 年的共识是 **Agentic 编排 + Hybrid 检索** 才是最佳组合，而非二选一。我们的 Chroma + bm25s + RRF 基础非常扎实，应在 Agentic 层复用。

---

## 5. 结论

Agentic RAG 不是对现有向量检索+BM25 方案的替代，而是**在其之上的智能编排层**。对于 Deep Research Agent 这类研究型应用，Agentic 改造的收益是明确的：

- **多跳推理召回率**可从 60-70% 提升至 85-90%（PRISM 数据）
- **端到端准确率**在复杂 QA 上提升 5-15%（HotpotQA 数据）
- **幻觉率**通过自验证机制显著降低

**推荐策略**：以最小侵入性的 **Phase 1（CRAG 式质量回退）** 为切入点，复用现有的 `batch_evaluate_sources` 和 `generate_broader_queries` 能力，在保持 194 测试通过的前提下逐步增强 Agentic 能力。

---

## 参考来源

1. [Agentic Retrieval-Augmented Generation: A Survey on Agentic RAG (arXiv:2501.09136)](https://arxiv.org/html/2501.09136v1)
2. [PRISM: Agentic Retrieval with LLMs for Multi-Hop QA (arXiv:2510.14278)](https://arxiv.org/abs/2510.14278)
3. [From BM25 to Agentic RAG: The Evolution of Machine Memory](https://interestingengineering.substack.com/p/from-bm25-to-agentic-rag-the-evolution)
4. [LangGraph Agentic RAG Official Example](https://github.com/langchain-ai/langgraph/blob/main/examples/rag/langgraph_agentic_rag.ipynb)
5. [LangChain Docs: Build a custom RAG agent with LangGraph](https://docs.langchain.com/oss/python/langgraph/agentic-rag)
6. [From BM25 to Corrective RAG: Benchmarking Retrieval Strategies (arXiv:2604.01733)](https://arxiv.org/html/2604.01733v1)
7. [A2RAG: Adaptive Agentic Graph Retrieval for Cost-Aware Reasoning (arXiv:2601.21162)](https://arxiv.org/abs/2601.21162)
8. [Agentic RAG for Dummies (GitHub)](https://github.com/GiovanniPasq/agentic-rag-for-dummies)
9. [RAG Paper Survey: From RETRO to Self-RAG and CRAG](https://www.youngju.dev/blog/ai-papers/2026-03-12-rag-retrieval-augmented-generation-self-rag-crag-survey.en)
10. [AdaGATE: Adaptive Gap-Aware Token-Efficient Evidence Assembly (arXiv:2605.05245)](https://arxiv.org/html/2605.05245v1)
