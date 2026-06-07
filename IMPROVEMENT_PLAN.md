# Deep Research Agent 改进计划 v1.0

**目标**: 将 DeepResearch-Bench Overall 分数从 55.66 提升至 62+，超越当前顶级水平（58.03）。

**核心策略**: 保持 Comprehensiveness & Insight 优势，重点突破 Instruction Following，优化 Readability。

---

## 当前差距分析

| 维度 | 当前 | 顶级 Agent | 差距 | 优先级 |
|------|------|------------|------|--------|
| Comprehensiveness | 55.43 | 53.87 | +1.56 ✅ | 保持 |
| Insight | 59.46 | 59.48 | -0.02 ✅ | 保持 |
| **Instruction Following** | **52.34** | **61.48** | **-9.14 ❌** | **P0** |
| Readability | 52.80 | ~55-58 | -2~5 ❌ | P1 |

**核心结论**: 系统在信息收集和分析方面已达顶级水平，但**对用户任务要求的回应能力严重不足**，这是最大突破口。

---

## Phase 1: Planner 任务解析增强（3-5 天）

### 1.1 任务要求提取

**问题**: Planner 只分解主题维度，未提取用户明确要求（比较/推荐/预测等）。

**改动**:
- 修改 `PLANNER` prompt，强制提取：
  - `core_objectives`: 用户想达成什么（比较/评估/推荐/预测）
  - `explicit_requirements`: 明确要求（"比较5个维度"、"推荐2-3家"）
  - `scope_constraints`: 范围限定（地域、时间、对象类型）
  - `sub_questions`: 所有子问题/子任务
- `ResearchState` 新增 `requirements` 字段
- `output_structure` 根据任务类型生成：
  - 比较任务 → 对比维度 + 综合对比表
  - 评估推荐 → 评估框架 + 候选分析 + 排名推荐
  - 预测任务 → 现状分析 + 驱动因素 + 明确预测

**验证**: 运行 5-10 个 benchmark 任务，检查 requirements 提取准确性。

### 1.2 任务要求传递至 Subtask

**改动**:
- `SPLITTER` prompt 要求每个 subtask 的 description 包含：
  - 需要回应的核心目标
  - 需要遵守的范围限定
  - 需要覆盖的具体子问题
- `run_subagent` 将 `requirements` 注入 system prompt

---

## Phase 2: Subagent 任务执行增强（5-7 天）

### 2.1 Subagent 任务意识

**改动**:
- `SUBAGENT_REPORT` prompt 增加 **TASK COMPLIANCE REQUIREMENTS**:
  - 必须直接回应核心目标
  - 如要求比较，必须提供显式对比分析
  - 如要求评估推荐，必须给出具体推荐和理由
  - 如要求预测，必须提供明确预测和支持证据
  - 严格遵守所有范围限定
- `SUBAGENT_REACT_SYSTEM` 增加自检提醒：
  - "每次搜索后检查是否正在回应用户原始任务"
  - "submit_report 前自我检查是否覆盖所有任务要求"

### 2.2 搜索策略优化

**改动**:
- `generate_search_queries` 根据任务类型添加修饰词：
  - 比较 → "对比"、"vs"、"comparison"
  - 评估 → "评估"、"排名"、"best"
  - 预测 → "预测"、"forecast"、"2025 2026"
- 根据 `scope_constraints` 添加范围限定词
- 增加中文搜索修饰词

### 2.3 Evidence 评估增强

**改动**:
- `SOURCE_EVALUATE` prompt 增加：
  - `task_relevance`: 对回应任务要求的重要性（0-3）
  - `requirement_coverage`: 覆盖了哪些任务要求
- `batch_evaluate_sources` 将 `task_relevance` 纳入 quality_score

---

## Phase 3: Synthesis 任务符合性保证（5-7 天）

### 3.1 Synthesis Prompt 增强

**改动**:
- `SYNTHESIS` prompt 增加 **PRE-WRITING CHECKLIST**:
  - [ ] 该 section 是否回应了用户核心目标？
  - [ ] 是否遵守了所有范围限定？
  - [ ] 如要求比较/评估/推荐/预测，是否包含？
  - [ ] 是否回答了所有子问题？
- 增加 **FINAL REPORT MUST INCLUDE**:
  - 必须包含直接回答主要问题的"结论与建议"section
  - 必须显式回应任何"比较/评估/推荐/预测"要求
  - 必须使用表格展示对比/评估结果

### 3.2 任务符合性后处理验证（关键！）

**改动**:
- 新增 `_verify_task_compliance(report, requirements, user_query)`:
  - 使用 LLM 检查报告是否满足所有任务要求
  - 识别缺失要求，生成 300-500 字补充内容
  - 在 `synthesize_report` 完成后调用
  - 支持 retry（最多 2 次）

### 3.3 报告格式优化

**改动**:
- 新增 `_format_report(report)` 后处理：
  - 清晰标题层级
  - 表格展示对比数据
  - 列表展示关键点
  - 加粗关键结论
  - 段落简洁（3-5 句）
  - Section 间过渡
- 优化引用格式和密度检查

---

## Phase 4: Reflection 迭代优化（3-5 天）

### 4.1 Reflection 检查细化

**改动**:
- `REFLECTION` prompt:
  - Threshold 从 0.6 → 0.7
  - 细化 `instruction_following` 检查：
    - 是否 DIRECTLY 回应核心目标？
    - 是否 STRICTLY 遵守范围限定？
    - 是否覆盖 ALL 子问题？
    - 比较/评估/推荐/预测是否 present and clear？
  - Gap 描述必须明确指出哪些任务要求未被满足
  - 新增 gap_type: `task_compliance`

### 4.2 迭代预算优化

**改动**:
- 配置新增：
  - `task_compliance_iterations`: 专门修复任务符合性的额外迭代（默认 1）
  - `max_iterations_total`: 总上限（默认 3）
- `graph.py`:
  - 正常 reflection 最多 2 次
  - 如有 task_compliance gap，增加 1 次专门修复
  - 保留原始 requirements 防止遗忘

---

## Phase 5: 中文与领域优化（3-5 天）

### 5.1 中文搜索优化

**改动**:
- `_SOURCE_MODIFIERS` 增加中文修饰词：
  - "中文": ["中文", "中国", "国内"]
  - "报告": ["研究报告", "行业报告"]
  - "数据": ["统计数据", "官方数据", "年报"]
  - "政策": ["政策", "法规", "政府文件"]
- SearXNG 中文引擎偏好
- Prompts 增加语言一致性检查

### 5.2 领域检测

**改动**:
- `planning.py` 根据 query 识别领域：
  - Finance → 偏好 industry report, official data
  - Tech → 偏好 academic, github
  - History → 偏好 academic, archives
- 领域特定 evaluation criteria

---

## Phase 6: Benchmark 迭代（7-10 天）

### 6.1 自动化 Benchmark

**改动**:
- 优化 `runner.py`：
  - 支持批量运行 100 任务
  - 断点续跑
  - 详细日志
- 创建 `analysis.py`：
  - 分析每个维度得分
  - 识别低分任务共同特征
  - 按领域/任务类型分类
  - 生成改进建议

### 6.2 迭代流程

1. 运行完整 benchmark（100 任务）
2. 分析低分任务（Overall < 50）
3. 针对性修改 prompts/逻辑
4. 重新运行验证
5. 重复 2-4 直到收敛

**预期迭代**: 2-3 轮

---

## 预期成果

| 维度 | 当前 | 完成后 | 提升 |
|------|------|--------|------|
| Comprehensiveness | 55.43 | 58 | +2.6 |
| Insight | 59.46 | 62 | +2.5 |
| **Instruction Following** | **52.34** | **62** | **+9.7** |
| Readability | 52.80 | 58 | +5.2 |
| **Overall** | **55.66** | **62+** | **+6.3** |

---

## 资源需求

- **LLM 调用**: 单次 benchmark 约 1000-2000 次
- **估算成本**: ~$5-15/次，完整迭代 ~$15-45
- **开发时间**: 3-4 周
- **验证时间**: 1-2 周

---

## 实施状态

| Phase | 状态 | 备注 |
|-------|------|------|
| Phase 1 | 🔄 进行中 | 切换模型 + 修改 Planner |
| Phase 2 | ⏳ 待开始 | |
| Phase 3 | ⏳ 待开始 | |
| Phase 4 | ⏳ 待开始 | |
| Phase 5 | ⏳ 待开始 | |
| Phase 6 | ⏳ 待开始 | |

---

## 关键决策记录

1. ✅ **优先级**: Instruction Following 为最高优先级
2. ✅ **模型**: 使用 kimi-k2.6（OpenRouter: moonshotai/kimi-k2-6）
3. ✅ **验证方式**: 先用 10-20 任务小样本子集快速验证
4. ✅ **架构改动**: 接受 ResearchState 新增 requirements 字段和后处理步骤
5. ✅ **时间**: 无约束，逐步优化
