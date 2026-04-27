# Deep Research Agent — 重构设计方案

## 一、产品定位

跨平台桌面应用（Windows + macOS），参考 Codex for Mac 的极简 UI 设计风格。

用户克隆仓库、配置 Python 环境和 API Key 后即可使用。一个窗口完成从输入研究主题到阅读最终报告的全部流程。

---

## 二、技术架构

```
┌──────────────────────────────────────────────────────┐
│                   Electron Shell                      │
│  ┌────────────────────────────────────────────────┐  │
│  │           Renderer Process (前端)                │  │
│  │                                                  │  │
│  │  HTML + CSS + Vanilla JS                        │  │
│  │  - 输入页（研究主题 + 参数配置）                   │  │
│  │  - 实时进度仪表盘（阶段、子 agent、来源）          │  │
│  │  - 报告阅读页（Markdown 渲染）                    │  │
│  │  - 设置页（API Key、模型选择）                    │  │
│  │  - 历史页（过往研究记录）                         │  │
│  └──────────────────┬─────────────────────────────┘  │
│                     │  HTTP + SSE (localhost)         │
│  ┌──────────────────▼─────────────────────────────┐  │
│  │         Python Backend (子进程)                  │  │
│  │                                                  │  │
│  │  FastAPI (localhost:随机端口)                    │  │
│  │  - LangGraph 研究流水线                          │  │
│  │  - 多 Provider LLM 路由（全异步）                 │  │
│  │  - 搜索层 (Firecrawl + DuckDuckGo fallback)      │  │
│  │  - SQLite 本地持久化                             │  │
│  └────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────┘
```

**Electron 主进程职责**：窗口管理、启动/关闭 Python 子进程

**Python 后端职责**：所有 AI 和数据处理逻辑，前端只做展示

**前后端通信**：Python 进程启动时选择随机空闲端口，写入临时文件；Electron 主进程读取端口号，前端通过 `fetch` 和 `EventSource` 连接 localhost。

---

## 三、原仓库可参考的思想

| 思想 | 来源 | 保留方式 |
|------|------|----------|
| 8 阶段 LangGraph 流水线 + reflection 循环 | `graph.py` | 重新实现，全异步 |
| 子 agent 并行 + 内部并行搜索/提取 | `agents.py:run_subagent` | asyncio 原生实现 |
| 源质量评分 + 领域多样性 + 自适应查询 | `agents.py` | 保留逻辑，优化批量效率 |
| 搜索回退链 Firecrawl → DuckDuckGo | `search.py` | 保留，增加 fallback 策略 |
| 多 Provider 统一抽象 | `providers/base.py` | 改为 async，简化接口 |
| 智能截断检测与续写 | `agents.py:_continue_if_truncated` | 保留思路，精简实现 |
| 按角色分配不同模型 | `config.py:AppConfig.roles` | 用 config.yaml 承载 |
| SSE 事件类型定义 | `server.py:_translate_event` | 继承事件类型，简化映射 |
| 三栏仪表盘布局 | `search-display.tsx` | 参考布局，Vanilla JS 实现 |
| 前端事件状态推导 | `search-display.tsx:useMemo` | 参考推导模式 |

---

## 四、目录结构

```
deep-research-agent/
├── package.json                    # Electron 项目 + 脚本定义
├── requirements.txt                # Python 依赖
├── config.yaml.example             # 配置文件模板
├── README.md
├── src/
│   ├── main/                       # Electron 主进程 (TypeScript)
│   │   ├── index.ts                # 窗口管理
│   │   └── python.ts               # Python 子进程管理（启动/停止/端口检测）
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
│       ├── server.py               # FastAPI 本地服务入口
│       ├── config.py               # 配置加载（config.yaml + 环境变量）
│       ├── models.py               # Pydantic 数据模型
│       ├── graph.py                # LangGraph 状态机构建
│       ├── agents.py               # 各阶段 agent 函数（plan/split/scale/subagent/reflection/synthesize/citation）
│       ├── prompts.py              # 系统提示词模板
│       ├── search.py               # 搜索层（Firecrawl + DDG fallback）
│       ├── providers/              # LLM Provider 适配器
│       │   ├── __init__.py         # Provider 注册与获取
│       │   ├── base.py             # 异步抽象基类（含重试逻辑）
│       │   ├── openai.py           # OpenAI (GPT-4o/o1)
│       │   ├── anthropic.py        # Anthropic Claude
│       │   ├── gemini.py           # Google Gemini (openai 兼容端点)
│       │   └── huggingface.py      # HuggingFace Inference API
│       ├── persistence.py          # SQLite 持久化（研究记录、checkpoint、来源）
│       └── export.py               # 报告导出（md/pdf）
└── tests/
    ├── conftest.py                 # pytest fixtures
    ├── test_config.py
    ├── test_agents.py
    ├── test_search.py
    ├── test_graph.py
    └── test_providers/
        ├── test_openai.py
        ├── test_anthropic.py
        └── test_gemini.py
```

---

## 五、UI 设计

核心风格：暗色主题、极简、大留白、清晰排版。参照 Codex 的视觉语言。

### 5.1 输入页（启动后默认显示）

```
┌──────────────────────────────────────────────────────┐
│  ● ● ●                                   Deep Research│
├──────────────────────────────────────────────────────┤
│                                                      │
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

### 5.2 研究进度页（点击开始研究后）

```
┌──────────────────────────────────────────────────────┐
│  ● ● ●                                   Deep Research│
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

### 5.3 报告阅读页（研究完成后）

```
┌──────────────────────────────────────────────────────┐
│  ● ● ●                                   Deep Research│
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
│  [导出 PDF]      │  ...                               │
│  [复制 Markdown] │                                    │
│  [新研究]        │                                    │
│                  │                                    │
└──────────────────┴────────────────────────────────────┘
```

### 5.4 设置页

```
┌──────────────────────────────────────────────────────┐
│  ● ● ●                                   Deep Research│
├──────────────────────────────────────────────────────┤
│                                                      │
│  设置                                                │
│  ──────────────────────────────────────────────       │
│  API Keys                                            │
│  Anthropic API Key    ●●●●●●●●●●●●●●●●  [显示]       │
│  OpenAI API Key       ●●●●●●●●●●●●●●●●  [显示]       │
│  Gemini API Key       ●●●●●●●●●●●●●●●●  [显示]       │
│  HF Token             ●●●●●●●●●●●●●●●●  [显示]       │
│  Firecrawl API Key    ●●●●●●●●●●●●●●●●  [显示]       │
│                                                      │
│  默认模型                                            │
│  Provider             [Anthropic ▼]                  │
│  Model                [Claude Sonnet 4 ▼]            │
│                                                      │
│  按角色分模型（可选）                                  │
│  Planner              [跟随默认 ▼]                    │
│  Subagent             [跟随默认 ▼]                    │
│  Coordinator          [跟随默认 ▼]                    │
│                                                      │
│  研究默认值                                          │
│  最大迭代深度          3                              │
│  来源质量阈值          0.7                            │
│                                                      │
└──────────────────────────────────────────────────────┘
```

---

## 六、Python 后端设计

### 6.1 依赖 (requirements.txt)

```
# Web (仅本地服务)
fastapi>=0.110
uvicorn[standard]>=0.30

# Config
pyyaml>=6.0
pydantic>=2.0
python-dotenv>=1.0

# LLM Provider SDK
openai>=1.50         # OpenAI + Gemini (兼容端点)
anthropic>=0.40      # Anthropic Claude

# Pipeline
langgraph>=0.2

# Search
firecrawl-py>=1.0    # 主搜索 + 内容提取
ddgs>=5.0            # DuckDuckGo 免费回退

# Persistence
db-sqlite3>=0.0      # 或直接用内置 sqlite3
```

### 6.2 核心改造：全异步

所有 Provider 和 Agent 函数改为原生 async，不再用 executor 线程：

```python
# providers/base.py
from abc import ABC, abstractmethod

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

# providers/anthropic.py
class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(self, api_key: str):
        self.client = anthropic.AsyncAnthropic(api_key=api_key)

    async def chat(self, model, messages, temperature=0.2, max_tokens=None):
        system, conversation = _split_system_messages(messages)
        response = await self.client.messages.create(
            model=model,
            system=system,
            messages=conversation,
            temperature=temperature,
            max_tokens=max_tokens or 8192,
        )
        return response.content[0].text
```

LangGraph 使用 `graph.ainvoke()` 替代 `graph.invoke()`。

### 6.3 配置系统

**优先级**：环境变量 > config.yaml > 默认值

配置文件位于 `~/.deep-research/config.yaml`，首次启动时自动从 `config.yaml.example` 生成：

```yaml
# ~/.deep-research/config.yaml

# 默认 LLM
default:
  provider: anthropic
  model: claude-sonnet-4-20250514

# 按角色分配模型（可选，未配置则跟随默认）
roles:
  planner:
    provider: openai
    model: gpt-4o
  subagent:
    provider: anthropic
    model: claude-haiku-3-5-20241022
  coordinator:
    provider: anthropic
    model: claude-sonnet-4-20250514
  reflection:
    provider: openai
    model: gpt-4o
  citation:
    provider: anthropic
    model: claude-haiku-3-5-20241022
  # splitter / scaler / evaluator 未配置则跟随默认

# 搜索配置
search:
  firecrawl_api_key: ${FIRECRAWL_API_KEY}

# 研究默认参数
research:
  max_iterations: 3
  quality_threshold: 0.7
  max_sources_per_domain: 3
  tool_calls_per_subagent: 15
```

API Key 推荐通过环境变量管理，config.yaml 中用 `${VAR}` 语法引用。设置页修改的配置直接写入 config.yaml。

### 6.4 持久化

本地 SQLite 数据库，位置 `~/.deep-research/history.db`：

```sql
-- 研究运行记录
CREATE TABLE runs (
    run_id TEXT PRIMARY KEY,
    query TEXT NOT NULL,
    status TEXT NOT NULL,           -- running | completed | cancelled | failed
    provider TEXT,
    model TEXT,
    config_snapshot TEXT,           -- JSON: 运行时完整配置快照
    total_sources INTEGER DEFAULT 0,
    total_reports INTEGER DEFAULT 0,
    iterations INTEGER DEFAULT 0,
    started_at INTEGER NOT NULL,
    completed_at INTEGER,
    report_path TEXT                -- 最终报告文件路径
);

-- 阶段检查点（支持后续研究恢复功能）
CREATE TABLE checkpoints (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    phase TEXT NOT NULL,
    state TEXT NOT NULL,            -- JSON: 完整 ResearchState
    created_at INTEGER NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

-- 发现的来源
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

-- 子 agent 报告
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

CREATE INDEX idx_runs_started_at ON runs(started_at DESC);
CREATE INDEX idx_checkpoints_run_id ON checkpoints(run_id);
CREATE INDEX idx_sources_run_id ON sources(run_id);
```

### 6.5 API 端点

仅监听 localhost，不暴露到网络：

| 端点 | 用途 |
|------|------|
| `POST /api/research` | 提交研究任务，返回 run_id |
| `GET /api/research/{run_id}/stream` | SSE 流，推送实时进度事件 |
| `POST /api/research/{run_id}/cancel` | 取消进行中的研究 |
| `GET /api/research/history` | 历史研究记录列表 |
| `GET /api/research/{run_id}/report` | 获取已完成研究的完整报告 |
| `GET /api/config` | 返回当前配置（脱敏，不含完整 API Key） |
| `POST /api/config` | 更新配置并写入 config.yaml |
| `GET /api/health` | 健康检查 |
| `GET /api/models` | 列出所有可用模型 |

### 6.6 SSE 事件类型

继承原仓库的事件类型，精简合并：

```
phase-update        { phase, message, run_id }
progress            { phase, percent }
plan-generated      { plan_preview, plan_length }
subtasks-created    { count, subtasks }
scaling-computed    { scaling }
subagents-launch    { iteration, total_agents, agent_details }
subagent-step       { subtask_id, subtask_title, step, message, evidence_count }
subagent-queries    { subtask_id, subtask_title, queries }
subagent-search     { subtask_id, query, status, results_found }
subagent-sources-scored { subtask_id, subtask_title, total_candidates, unique_sources, top_urls, top_scores }
subagent-extract    { subtask_id, url, status }
subagent-complete   { subtask_id, subtask_title, report_length, sources_count, evidence_count }
llm-call            { status, model, provider, role, attempt, output_length, error }
reflection-decision { decision, iteration, new_subtask_count, new_subtasks, total_reports, total_sources }
report-draft        { content, report_length }
citations-added     { cited_report_length }
final-result        { content }
complete            { run_id, total_sources, total_reports, iterations, provider, model }
error               { error, phase, hint }
warning             { phase, message }
```

---

## 七、实施计划

### Milestone 1：核心引擎（5-7 天）

**目标**：Python 研究引擎能独立跑通全流程，不依赖 Electron

- [ ] 项目初始化：`requirements.txt`、目录结构
- [ ] 配置系统：config.yaml 加载 + 环境变量覆盖 + 默认值
- [ ] Provider 层：`LLMProvider` async 基类 + Anthropic / OpenAI / Gemini / HuggingFace 实现，含重试逻辑
- [ ] 搜索层：Firecrawl 主搜索 + DuckDuckGo 免费回退，统一返回格式
- [ ] LangGraph 流水线：8 阶段全流程（graph.ainvoke）
- [ ] 子 agent 并行 + 源质量评分 + 领域多样性 + 自适应查询优化
- [ ] 智能截断检测与续写
- [ ] SQLite 持久化（checkpoint 写入、报告存储）
- [ ] FastAPI 服务 + 全部 API 端点 + SSE 流
- [ ] pytest 单元测试（providers / search / agents / graph）

**验收**：`python -m src.backend.server` 启动，curl 提交研究后能通过 SSE 接收进度事件并拿到完整报告

### Milestone 2：Electron 壳 + 输入页 + 报告页（3-4 天）

**目标**：桌面应用框架跑通，能提交研究并看到结果

- [ ] Electron 项目初始化（package.json、tsconfig、窗口配置）
- [ ] 主进程：无边框窗口 + 自定义标题栏（macOS traffic lights 适配）
- [ ] Python 子进程管理（启动时随机端口、stdout/stderr 管道、进程退出时清理）
- [ ] 暗色主题 + 全局 CSS 变量 + 排版系统
- [ ] 输入页：研究主题输入框、深度选择按钮、模型下拉菜单
- [ ] 前端 API 层：封装 fetch（普通请求）+ EventSource（SSE 流）
- [ ] 报告页：marked.js Markdown 渲染 + highlight.js 代码高亮
- [ ] 设置页：API Key 输入（密码掩码 + 显示/隐藏切换）、模型选择
- [ ] 页面间导航（输入 → 进度 → 报告）

**验收**：桌面窗口打开，输入研究主题后能看到最终报告渲染

### Milestone 3：实时进度仪表盘（5-7 天）

**目标**：研究过程中能看到所有阶段和子 agent 的实时状态

- [ ] SSE 事件类型定义（前端 side，与后端对齐）
- [ ] `store.js`：事件 → 状态推导（参考原仓库 useMemo 模式），驱动所有组件渲染
- [ ] 阶段时间线组件：8 个阶段的实时状态（pending/active/completed），动画进度条
- [ ] 子 agent 并行面板：每个 agent 卡片展示状态、查询词、搜索次数、来源数、提取进度
- [ ] 来源面板：质量评分条（颜色分段）、领域分布标签、实时追加动画
- [ ] 左侧边栏：来源面板 / 代理面板 / 活动日志 三 tab 切换
- [ ] 进度页 → 报告页无缝切换（研究完成后自动跳转，保留过程可回看）
- [ ] 取消研究：发送 cancel 请求，清理 UI 状态
- [ ] 窗口关闭确认（研究进行中时弹出确认对话框）

**验收**：提交研究后，所有阶段和子 agent 状态实时可见，布局与第五章设计稿一致

### Milestone 4：设置 + 历史 + 导出（2-3 天）

**目标**：完整的产品体验闭环

- [ ] 设置页完善：读取/写入 config.yaml，API Key 持久化存储，按角色模型配置
- [ ] 历史页：列表展示过往研究（问题、时间、来源数、报告数），点击可重新查看完整报告
- [ ] 报告导出：Markdown 文件保存到本地、通过浏览器打印生成 PDF
- [ ] 错误处理：API Key 缺失/无效提示、网络连接失败提示、LLM 调用失败提示（含原仓库的 `_get_error_hint` 风格指引）
- [ ] 首次启动引导：检测 config.yaml 是否存在，不存在则引导用户完成初始化配置
- [ ] 报告中引用链接可点击（用系统默认浏览器打开）

**验收**：从首次启动 → 配置 → 输入主题 → 实时进度 → 查看报告 → 查看历史 → 导出，完整链路走通

---

## 八、v1 范围与决策记录

**v1 范围**：Milestone 1-4，不包含打包发布。

**技术决策**：

| 决策项 | 选择 | 原因 |
|--------|------|------|
| 前端框架 | Vanilla JS（不用 React） | 页面交互不复杂，减少依赖和构建步骤 |
| Python 分发方式 | `.py` 源码 + `requirements.txt` | v1 不打包，用户自行配置 Python 环境 |
| Provider 模型 | 全异步 (async/await) | 避免 executor 线程，提升 I/O 密集型并发 |
| 持久化 | SQLite (关键路径) | 原仓库 Supabase best-effort 不可靠 |
| LLM Provider SDK | 各用原生 SDK (anthropic, openai) | 获取最新特性，不通过统一 openai 兼容层 |
| Gemini 接入方式 | openai SDK (兼容端点) | 减少依赖，统一代码路径 |
| 报告渲染 | marked.js + highlight.js | 轻量，足够覆盖 Markdown + 代码块 |

**后续版本考虑**：
- Milestone 5：electron-builder 打包发布（.dmg + .exe）
- 研究恢复：从 checkpoint 恢复中断的研究
- 多语言报告（非英语 query 的 prompt 优化）
- 更多搜索 provider（Tavily, SerpAPI）

---

## 九、实施状态

| Milestone | 内容 | 状态 |
|-----------|------|------|
| M1 | 核心引擎 | 🔲 待开始 |
| M2 | Electron 壳 + 输入页 + 报告页 | 🔲 待开始 |
| M3 | 实时进度仪表盘 | 🔲 待开始 |
| M4 | 设置 + 历史 + 导出 | 🔲 待开始 |
| M5 | 打包发布 | ⏸️ 延后 |
