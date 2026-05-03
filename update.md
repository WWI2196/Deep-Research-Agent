# Deep Research Agent — 架构设计

## 一、产品定位

Web 应用（Python FastAPI 后端 + 浏览器前端），参考 Codex for Mac 的极简 UI 设计风格。

用户克隆仓库、配置 Python 环境和 API Key 后即可使用。浏览器中完成从输入研究主题到阅读最终报告的全部流程。

---

## 二、技术架构

```
┌──────────────────────────────────────────────────────┐
│                   Browser (http://127.0.0.1:8787)     │
│  ┌────────────────────────────────────────────────┐  │
│  │           Frontend (Vanilla JS)                  │  │
│  │                                                  │  │
│  │  HTML + CSS + Vanilla JS                        │  │
│  │  - 输入页（研究主题 + 参数配置）                   │  │
│  │  - 实时进度仪表盘（阶段、子 agent、来源）          │  │
│  │  - 报告阅读页（Markdown 渲染）                    │  │
│  │  - 设置页（配置管理）                             │  │
│  │  - 历史页（过往研究记录）                         │  │
│  └──────────────────┬─────────────────────────────┘  │
│                     │  HTTP + SSE (same-origin)       │
│  ┌──────────────────▼─────────────────────────────┐  │
│  │         Python Backend (FastAPI)                 │  │
│  │                                                  │  │
│  │  FastAPI (127.0.0.1:8787)                       │  │
│  │  - LangGraph 研究流水线 (8 节点)                  │  │
│  │  - 多 Provider LLM 路由（全异步）                 │  │
│  │  - 搜索层 (SearXNG + trafilatura 提取)           │  │
│  │  - SQLite 本地持久化                             │  │
│  │  - 静态文件服务 (前端)                            │  │
│  └────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────┘
        │
        ▼
┌──────────┐  ┌──────────┐
│ SearXNG  │  │ LLM API  │
│ :8080    │  │ 6 built-in│
│ 70+ eng. │  │ providers│
│ free     │  │          │
└──────────┘  └──────────┘
```

**后端职责**：所有 AI 和数据处理逻辑，同时提供静态文件服务。

**前端职责**：纯展示层，通过 `fetch` 和 SSE 与后端通信。

---

## 三、原仓库可参考的思想

| 思想 | 来源 | 实现方式 |
|------|------|----------|
| 8 阶段 LangGraph 流水线 + reflection 循环 | `graph.py` | 全异步 `graph.ainvoke()` |
| 子 agent 并行 + 内部并行搜索/提取 | `agents.py:run_subagent` | `asyncio.gather` 原生实现 |
| 源质量评分 + 领域多样性 + 自适应查询 | `agents.py` | 批量评分、领域上限 3 条 |
| SearXNG 自托管搜索 (70+ 引擎) | `search.py` | JSON API，无需 API Key |
| trafilatura 内容提取 | `search.py` | 免费，干净 markdown 输出 |
| 多 Provider 统一抽象 | `providers/base.py` | 全异步 + 指数退避重试 |
| 智能截断检测与续写 | `agents.py:_continue_if_truncated` | 检测悬挂连接词 |
| 按角色分配不同模型 | `config.py:AppConfig.roles` | config.yaml 承载 |
| SSE 事件类型定义 | `server.py:_translate_event` | 16 种事件类型 |
| 前端事件状态推导 | `store.js` | fetch + SSE 驱动渲染 |

---

## 四、目录结构

```
deep-research-agent/
├── pyproject.toml                  # Python 项目 + 依赖 (uv)
├── config.yaml.example             # 配置文件模板
├── README.md
├── CLAUDE.md
├── src/
│   ├── renderer/                   # 前端 UI (Vanilla JS)
│   │   ├── index.html              # 主页面
│   │   ├── css/
│   │   │   └── style.css           # 全局样式 + 暗色主题变量
│   │   ├── js/
│   │   │   ├── app.js              # 应用入口、页面路由
│   │   │   ├── api.js              # HTTP + SSE 通信层
│   │   │   ├── store.js            # 全局状态管理（事件 → 状态推导）
│   │   │   ├── components/
│   │   │   │   ├── input.js        # 研究输入页
│   │   │   │   ├── dashboard.js    # 实时进度仪表盘
│   │   │   │   ├── phases.js       # 阶段时间线 + 进度条
│   │   │   │   ├── subagents.js    # 子 agent 并行面板
│   │   │   │   ├── sources.js      # 来源面板（质量评分、领域分布）
│   │   │   │   ├── report.js       # 报告阅读页
│   │   │   │   ├── settings.js     # 设置页
│   │   │   │   └── history.js      # 历史记录页
│   │   │   └── utils/
│   │   │       ├── markdown.js     # Markdown 渲染（marked.js）
│   │   │       └── format.js       # 格式化工具
│   │   └── assets/
│   │       └── icon.svg
│   └── backend/                    # Python 研究引擎
│       ├── __init__.py
│       ├── server.py               # FastAPI 服务入口（含静态文件服务）
│       ├── config.py               # 配置加载（config.yaml + 环境变量）
│       ├── models.py               # ResearchState TypedDict + Pydantic 模型
│       ├── graph.py                # LangGraph 状态机构建
│       ├── agents.py               # 各阶段 agent 函数
│       ├── prompts.py              # 系统提示词模板
│       ├── search.py               # SearXNG 搜索 + trafilatura 提取
│       ├── providers/              # LLM Provider 适配器
│       │   ├── __init__.py         # Provider 工厂注册
│       │   ├── base.py             # 异步抽象基类（含重试逻辑）
│       │   ├── openai_compatible.py # OpenAI 兼容 provider
│       │   └── anthropic.py        # Anthropic provider
│       ├── persistence.py          # SQLite 持久化
│       └── export.py               # 报告导出 (Markdown)
└── tests/
    ├── conftest.py                 # pytest fixtures
    ├── test_config.py
    ├── test_agents.py
    ├── test_search.py
    ├── test_graph.py
    ├── test_server.py
    ├── test_persistence.py
    ├── test_models.py
    ├── test_export.py
    ├── test_integration.py
    └── test_providers/
        ├── test_providers.py
        ├── test_anthropic.py
        └── test_openai_compatible.py
```

---

## 五、UI 设计

核心风格：暗色主题、极简、大留白、清晰排版。参照 Codex 的视觉语言。

### 5.1 输入页（启动后默认显示）

```
┌──────────────────────────────────────────────────────┐
│  Deep Research                                       │
├──────────────────────────────────────────────────────┤
│                                                      │
│              探索任何主题的深度研究                      │
│                                                      │
│  ┌──────────────────────────────────────────────────┐│
│  │                                                  ││
│  │  输入你想研究的问题...                             ││
│  │                                                  ││
│  └──────────────────────────────────────────────────┘│
│                                                      │
│  深度: [1] [2] [3]    模型: [Claude Sonnet 4 ▼]      │
│                                                      │
│              [ 开始研究 ]                             │
│                                                      │
│  ────────────────────────────────────────────────     │
│  最近研究                                            │
│  ├─ AI 对软件工程的影响      12min ago  26 sources   │
│  ├─ 量子计算最新突破         2h ago     18 sources   │
│  └─ 微塑料的环境影响         1d ago     31 sources   │
│                                                      │
└──────────────────────────────────────────────────────┘
```

### 5.2 研究进度页

```
┌──────────────────────────────────────────────────────┐
│  Deep Research                                       │
├──────────────────┬────────────────────────────────────┤
│                  │                                    │
│  ◉ 来源 (12)     │  研究进度                          │
│                  │  ████████████░░░░░░  68%           │
│  nature.com 92%  │                                    │
│  arxiv.org  88%  │  ✅ Plan      研究计划生成完成       │
│  github.io  85%  │  ✅ Split     5个子任务已创建       │
│  mit.edu    78%  │  ✅ Scale     复杂度：中等          │
│  ieee.org   72%  │  ⟳ Agents    3/5 完成              │
│  ...             │    ├ ✅ 市场格局    12源, 3.2k     │
│                  │    ├ ✅ 技术评估    8源, 2.8k      │
│  ──────────────  │    ├ ✅ 关键玩家    6源, 1.9k      │
│  领域分布        │    ├ ⟳ 监管分析    搜索中...       │
│  ┌─ tech 5      │    └ ⏳ 未来展望    排队中          │
│  ┌─ academic 4   │  ⏳ Reflection                      │
│  ┌─ news 3       │  ⏳ Synthesize                      │
│                  │  ⏳ Citations                       │
│  ◉ 代理 (5)      │                                    │
│  初始(3) 缺口(2) │  LLM调用: 8 | 耗时: 2m34s          │
│                  │                                    │
└──────────────────┴────────────────────────────────────┘
```

### 5.3 报告阅读页

```
┌──────────────────────────────────────────────────────┐
│  Deep Research                                       │
├──────────────────┬────────────────────────────────────┤
│                  │                                    │
│  研究过程 ▸      │  # AI 对软件工程的影响              │
│                  │                                    │
│  ◉ 来源 (26)     │  ## 摘要                           │
│                  │  人工智能正在深刻改变软件工程的...   │
│                  │                                    │
│  完成统计        │  ## 1. 市场格局分析                 │
│  26 来源         │  ...(2,100 words)...               │
│  5 报告          │                                    │
│  2 轮迭代        │  ## 2. 技术演进趋势                 │
│  耗时 2m34s      │  ...(1,800 words)...               │
│                  │                                    │
│  ──────────────  │  ## 结论                           │
│  [导出 Markdown] │  ...                               │
│  [复制]          │                                    │
│  [新研究]        │                                    │
│                  │                                    │
└──────────────────┴────────────────────────────────────┘
```

### 5.4 设置页

```
┌──────────────────────────────────────────────────────┐
│  Deep Research                                       │
├──────────────────────────────────────────────────────┤
│                                                      │
│  设置                                                │
│  ──────────────────────────────────────────────       │
│  API Keys                                            │
│  Anthropic API Key    ●●●●●●●●●●●●●●●●  [显示]       │
│  OpenAI API Key       ●●●●●●●●●●●●●●●●  [显示]       │
│  MIMO API Key         ●●●●●●●●●●●●●●●●  [显示]       │
│                                                      │
│  默认模型                                            │
│  Provider             [mimo ▼]                       │
│  Model                [mimo-v2.5 ▼]                  │
│                                                      │
│  按角色分模型（可选）                                  │
│  Planner              [跟随默认 ▼]                    │
│  Subagent             [跟随默认 ▼]                    │
│  Coordinator          [跟随默认 ▼]                    │
│  Reflection           [跟随默认 ▼]                    │
│                                                      │
│  研究默认值                                          │
│  最大迭代深度          3                              │
│  来源质量阈值          0.7                            │
│                                                      │
└──────────────────────────────────────────────────────┘
```

---

## 六、Python 后端设计

### 6.1 依赖 (pyproject.toml + uv)

```
fastapi>=0.110
uvicorn[standard]>=0.30
pyyaml>=6.0
pydantic>=2.0
python-dotenv>=1.0
openai>=1.50
anthropic>=0.40
langgraph>=0.2
trafilatura>=2.0
```

### 6.2 全异步 Provider

所有 Provider 和 Agent 函数基于原生 async/await：

```python
# providers/base.py
class LLMProvider(ABC):
    name: str

    @abstractmethod
    async def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> str: ...

    async def chat_with_retry(
        self, *args, max_retries: int = 3, **kwargs
    ) -> str:
        for attempt in range(max_retries):
            try:
                return await self.chat(*args, **kwargs)
            except Exception as exc:
                if _is_fatal(exc) or attempt == max_retries - 1:
                    raise
                await asyncio.sleep(2 ** attempt)
```

LangGraph 使用 `graph.ainvoke()` 替代 `graph.invoke()`。

### 6.3 配置系统

**优先级**：环境变量 > config.yaml > 内置默认值

配置文件位于 `~/.deep-research/config.yaml`，首次启动时参考 `config.yaml.example`：

```yaml
# ~/.deep-research/config.yaml

providers:
  mimo:
    api_key: your-mimo-api-key-here

default:
  provider: mimo
  model: mimo-v2.5

roles:
  planner: { provider: mimo, model: mimo-v2.5-pro }
  subagent: { provider: mimo, model: mimo-v2.5-pro }
  coordinator: { provider: mimo, model: mimo-v2.5-pro }
  reflection: { provider: mimo, model: mimo-v2.5-pro }

research:
  max_iterations: 3
  quality_threshold: 0.7
  max_sources_per_domain: 3
  tool_calls_per_subagent: 15
```

6 个内置 provider：`mimo`, `openai`, `anthropic`, `gemini`, `deepseek`, `openrouter`。支持 `${VAR}` 环境变量替换。可通过 `type: openai` 添加自定义 OpenAI 兼容服务。

### 6.4 持久化

本地 SQLite 数据库 `~/.deep-research/history.db`：

```sql
CREATE TABLE runs (
    run_id TEXT PRIMARY KEY,
    query TEXT NOT NULL,
    status TEXT NOT NULL,
    provider TEXT,
    model TEXT,
    config_snapshot TEXT,
    total_sources INTEGER DEFAULT 0,
    total_reports INTEGER DEFAULT 0,
    iterations INTEGER DEFAULT 0,
    started_at INTEGER NOT NULL,
    completed_at INTEGER,
    report_path TEXT
);

CREATE TABLE checkpoints (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    phase TEXT NOT NULL,
    state TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

CREATE TABLE sources (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    url TEXT NOT NULL,
    title TEXT,
    quality_score REAL,
    domain TEXT,
    subtask_id TEXT,
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

CREATE TABLE subagent_reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    subtask_id TEXT NOT NULL,
    content TEXT NOT NULL,
    sources_count INTEGER,
    evidence_count INTEGER,
    created_at INTEGER NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);
```

### 6.5 API 端点

仅监听 localhost：

| 端点 | 用途 |
|------|------|
| `GET /api/health` | 健康检查 + provider 信息 |
| `GET /api/config` | 当前配置（API Key 脱敏） |
| `POST /api/config` | 更新配置并写入 config.yaml |
| `GET /api/models` | 列出所有可用 provider |
| `POST /api/research` | 提交研究任务，返回 run_id |
| `POST /api/research/stream` | 研究任务 + SSE 实时流 |
| `POST /api/research/{id}/cancel` | 取消进行中的研究 |
| `GET /api/research/history` | 历史研究记录列表 |
| `GET /api/research/{id}/report` | 获取已完成研究的完整报告 |

### 6.6 SSE 事件类型

```
phase-update            { phase, message, run_id }
progress                { phase, percent }
plan-generated          { plan_preview, plan_length }
subtasks-created        { count, subtasks }
scaling-computed        { scaling }
subagents-launch        { iteration, total_agents, agent_details }
subagent-step           { subtask_id, subtask_title, step, message, evidence_count }
subagent-queries        { subtask_id, subtask_title, queries }
subagent-search         { subtask_id, query, status, results_found }
subagent-sources-scored { subtask_id, subtask_title, total_candidates, unique_sources, top_urls, top_scores }
subagent-extract        { subtask_id, url, status }
subagent-complete       { subtask_id, subtask_title, report_length, sources_count, evidence_count }
llm-call                { status, model, provider, role, attempt, output_length, error }
reflection-decision     { decision, iteration, new_subtask_count, new_subtasks, total_reports, total_sources }
report-draft            { content, report_length }
citations-added         { cited_report_length }
final-result            { content }
complete                { run_id, total_sources, total_reports, iterations, provider, model }
error                   { error, phase, hint }
warning                 { phase, message }
```

---

## 七、技术决策

| 决策项 | 选择 | 原因 |
|--------|------|------|
| 前端框架 | Vanilla JS（不用 React） | 页面交互不复杂，减少依赖和构建步骤 |
| 后端 | Python FastAPI + LangGraph | 全异步 pipeline，成熟生态 |
| 搜索 | SearXNG (自托管) | 70+ 引擎聚合，免费无限制 |
| 内容提取 | trafilatura | 免费 pip 安装，输出干净 markdown |
| Provider 模型 | 全异步 (async/await) | 提升 I/O 密集型并发性能 |
| 持久化 | SQLite | 本地存储，零依赖 |
| 报告渲染 | marked.js (CDN) | 轻量，足够覆盖 Markdown + 代码块 |
| 包管理 | uv (pyproject.toml) | 快速、现代的 Python 包管理器 |

---

## 八、实施状态

| 模块 | 内容 | 状态 |
|------|------|------|
| 配置系统 | config.yaml + env + 内置默认值，6 provider | ✅ 完成 |
| Provider 层 | async 基类 + Anthropic + OpenAI 兼容，retry 逻辑 | ✅ 完成 |
| 搜索层 | SearXNG 搜索 + trafilatura 提取 | ✅ 完成 |
| LangGraph 流水线 | 8 阶段全流程 (graph.ainvoke) | ✅ 完成 |
| 子 agent 并行 | asyncio.gather + 源质量评分 + 领域多样性 | ✅ 完成 |
| 持久化 | SQLite (runs, checkpoints, sources, subagent_reports) | ✅ 完成 |
| FastAPI 服务 | 9 端点 + SSE streaming + 静态文件服务 | ✅ 完成 |
| 前端 UI | 5 页面 (输入/进度/报告/历史/设置) | ✅ 完成 |
| 测试 | 160 tests, 0 failed | ✅ 完成 |
