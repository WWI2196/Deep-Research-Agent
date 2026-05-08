# Deep Research Agent — 文档库 RAG 集成方案

> 版本：v1.2（已完成）
> 日期：2026-05-07
> 状态：**全部四阶段已实现并测试通过**

---

## 一、需求拆解

### 1.1 核心目标
在现有 Deep Research Agent 中引入**私有文档库 RAG**能力，使用户能够：
1. **管理文档库** — 创建/删除/重命名文档库，上传/删除/分类文档
2. **研究时勾选检索** — 发起研究请求时选择要检索的一个或多个文档库
3. **可信优先级** — 文档库检索到的内容默认比网络搜索结果更可信，在引用和综合时赋予更高权重

### 1.2 功能需求矩阵

| 模块 | 功能 | 优先级 | 状态 |
|------|------|--------|------|
| **文档库管理** | 创建/删除/重命名文档库（Collection） | P0 | ✅ 完成 |
| | 上传文档（PDF/DOCX/TXT/MD/HTML） | P0 | ✅ 完成（异步解析 + pending 状态） |
| | 文档分类/标签（Category/Tag） | P1 | ✅ 完成（category 字段 + metadata 过滤） |
| | 文档列表/搜索/删除 | P0 | ✅ 完成 |
| | 重新索引（某文档或全库） | P1 | ✅ 完成（API + 前端 ↻ 按钮） |
| **RAG 检索** | 语义检索（向量相似度 Top-K） | P0 | ✅ 完成 |
| | **混合检索（向量 + BM25 + RRF）** | P0 | ✅ 完成 |
| | 检索结果重排序（Cross-encoder） | P2 | ⏸ 未实现（P2 需求） |
| **Pipeline 集成** | Subagent 阶段并行检索文档库 | P0 | ✅ 完成 |
| | 文档来源标记 + 高可信分数 | P0 | ✅ 完成（source: "document", score >= 0.85） |
| | Citation 阶段兼容 `[src: url]` 格式 | P0 | ✅ 完成（file:// 路径 + Path.exists() 验证） |
| | Synthesis 阶段文档证据优先引用 | P1 | ✅ 完成（prompt 中强调保留 file:// 引用） |
| **前端** | 研究输入页增加"文档库"勾选区 | P0 | ✅ 完成 |
| | 文档库管理页面 | P0 | ✅ 完成（/library 路由） |
| | 上传进度/解析状态展示 | P1 | ✅ 完成（pending/indexing/indexed/failed + 轮询刷新） |
| | 引用来源区分网络/文档图标 | P1 | ✅ 完成（📄 vs 🌐） |

### 1.3 非功能需求

| 维度 | 要求 | 状态 |
|------|------|------|
| **本地化** | 全部数据本地存储，不依赖第三方云向量库 | ✅ Chroma 文件持久化 |
| **免费优先** | Embedding 模型、向量存储均使用开源/免费方案 | ✅ bge-small-zh + bm25s |
| **性能** | 文档解析 + 向量化异步执行，不阻塞 API；检索延迟 < 500ms（单机） | ✅ 异步 indexing + 实测 < 300ms |
| **可扩展** | 设计预留多用户隔离接口（当前按单用户实现，DB 层预留 user_id） | ✅ 结构预留 |
| **兼容** | 不破坏现有 171+49 测试；不改动现有 SearXNG 搜索链路 | ✅ 179+49 测试通过 |

---

## 二、技术选型

### 2.1 方案总览

| 组件 | 选型 | 理由 |
|------|------|------|
| **向量数据库** | **Chroma** | 纯 Python、pip 即装、文件持久化、Collection 概念天然对应文档库 |
| **Embedding 模型** | **BAAI/bge-small-zh-v1.5** | 384维（与 MiniLM 同尺寸），~50MB，中文适配最优，CPU 推理快，Apache-2.0 |
| **关键词检索** | **bm25s** + **jieba** | bm25s 是纯 Python 最快 BM25 实现（比 rank-bm25 快 500 倍）；jieba 中文分词标准库，纯 Python |
| **融合策略** | **RRF (Reciprocal Rank Fusion)** | 无参数调优负担，对向量/BM25 两组排序结果稳健融合 |
| **文本分块** | **RecursiveCharacterTextSplitter** | 按语义边界切分，中文适配（自定义分隔符：。！？等） |
| **文档解析** | **pdfplumber** + **python-docx** + **trafilatura** + 原生 TXT/MD | 保留段落结构，复用现有 trafilatura |
| **前端** | **Vanilla JS**（不引入框架） | 如无必要勿增实体，现有 Store + 路由机制足够 |

### 2.2 Embedding 模型对比

| 指标 | all-MiniLM-L6-v2 | **bge-small-zh-v1.5** | bge-base-zh-v1.5 |
|------|------------------|-----------------------|------------------|
| 维度 | 384 | **384** | 768 |
| 模型大小 | ~50MB | **~50MB** | ~100MB |
| 中文效果 | 一般（训练语料英文为主） | **优秀**（专为中文优化） | 更优 |
| 推理速度 | ~3000 sent/sec (CPU) | **~2500 sent/sec (CPU)** | ~1200 sent/sec |
| 许可 | Apache-2.0 | **Apache-2.0** | Apache-2.0 |

---

## 三、架构设计

### 3.1 数据模型

```
┌─────────────────────────────────────────────────────────────┐
│                        Collection (文档库)                    │
├─────────────┬───────────────────────────────────────────────┤
│ id          │ UUID                                          │
│ name        │ 用户可见名称（如"AI 论文库"）                    │
│ description │ 描述                                          │
│ created_at  │ timestamp                                     │
│ doc_count   │ 文档数量（SQL COUNT 实时计算）                   │
│ status      │ ready / indexing / error                      │
└─────────────┴───────────────────────────────────────────────┘
                              │ 1:N
┌─────────────────────────────────────────────────────────────┐
│                        Document (文档)                        │
├─────────────┬───────────────────────────────────────────────┤
│ id          │ UUID                                          │
│ collection_id│ FK -> Collection                             │
│ name        │ 原始文件名                                     │
│ file_path   │ 本地存储路径（~/.deep-research/docs/）          │
│ file_type   │ pdf / docx / txt / md / html                  │
│ category    │ 用户定义分类（可选）                            │
│ tags        │ JSON 数组                                      │
│ page_count  │ 页数/大致规模                                  │
│ chunk_count │ 切分后的块数                                   │
│ status      │ pending / indexing / indexed / failed         │
│ error_msg   │ 失败原因                                       │
│ created_at  │ timestamp                                     │
└─────────────┴───────────────────────────────────────────────┘
```

**存储分层**：
- **SQLite**（`history.db`）：`collections`、`documents` 元数据表
- **Chroma**（`~/.deep-research/chroma/`）：向量 + chunk text + metadata
- **bm25s**（`~/.deep-research/bm25/{collection_id}/`）：每个文档库独立一个 BM25 索引目录
- **原始文件**（`~/.deep-research/docs/{collection_id}/{doc_id}.{ext}`）：本地磁盘

### 3.2 混合检索流程

```
用户查询/子任务目标
       │
       ├─► [Chroma 向量检索] ──► Top-K 向量结果（含相似度分数）
       │                            │
       ├─► [bm25s 关键词检索] ──► Top-K BM25 结果（含排名）
       │                            │
       ▼                            ▼
              [RRF 融合] ──► 并集去重 ──► 按 RRF_score 排序
                            │
                            ▼
                    最终 Top-K chunks
                            │
                            ▼
              包装为 evidence ──► 注入 subagent report
```

**关键修复记录**：
1. **Chroma 维度不匹配**：`collection.query(query_texts=...)` 会使用默认 embedding 函数（384维 ONNX MiniLM），与 bge-small-zh（512维）冲突。修复：手动 encode query，传入 `query_embeddings` 参数。
2. **SQLite 同名列覆盖**：`SELECT c.*, COUNT(d.id) as doc_count` 中 `c.*` 的 `doc_count` 列（默认 0）覆盖了 `COUNT` 结果。修复：显式列名替换 `c.*`。
3. **BM25 k 值超限**：`top_k * 2` 可能超过 corpus size。修复：`min(top_k * 2, corpus_size)`。

### 3.3 Pipeline 集成

```
init → plan → split → subagents ┬→ [SearXNG 搜索] ──┐
                                 ├→ [文档库混合检索] ─┤→ 合并 evidence → reflection → synthesize → cite
                                 └→ [文档来源高权重]  │
```

**文档来源处理**：
- `_search_document_collections()` 返回 `source: "document"`, `url: "file://..."`
- `normalize_search_item()` 保留原始 `source` 字段（修复前被覆盖为 `"search"`）
- `batch_evaluate_sources` 后保底 `quality_score = max(..., 0.85)`, `full_text = True`
- `_extract_one()` 对 `file://` 跳过 trafilatura，直接返回 `None`
- evidence 构建时使用 `[DOCUMENT] chunk_text` 格式
- `enforce_source_diversity` 对 `file://` URL 用文件父路径作为 domain（修复前空字符串 domain 导致同一库文档被限制为 3 条）

### 3.4 API 端点

| Method | Path | 描述 |
|--------|------|------|
| `GET` | `/api/collections` | 列出所有文档库 |
| `POST` | `/api/collections` | 创建文档库 `{name, description}` |
| `DELETE` | `/api/collections/{id}` | 删除文档库（级联删除文档、向量、BM25 索引） |
| `PATCH` | `/api/collections/{id}` | 重命名/修改描述 |
| `GET` | `/api/collections/{id}/documents` | 列出库内文档 |
| `POST` | `/api/collections/{id}/documents` | 上传文档，立即返回 pending |
| `DELETE` | `/api/collections/{id}/documents/{doc_id}` | 删除文档 |
| `GET` | `/api/collections/{id}/documents/{doc_id}/download` | 下载原始文件 |
| `POST` | `/api/collections/{id}/search` | 在指定库内混合检索 |
| `POST` | `/api/collections/{id}/reindex` | 全库重新索引 |
| `POST` | `/api/collections/{id}/documents/{doc_id}/reindex` | 单文档重新索引 |

### 3.5 前端

- **Library 页面** (`/library`)：左侧文档库列表，右侧文档列表。支持创建/删除/上传/重索引。文档状态显示：⏳ pending / ⟳ indexing / ✓ indexed / ✗ failed。2 秒轮询自动刷新。
- **Input 页面**：输入框下方"Document Libraries"折叠面板，复选框选择库，已选库以 tag 展示。
- **Sources 面板**：`source === "document"` 显示 📄 图标，`source === "searxng"` 显示 🌐 图标。

---

## 四、实现记录

### Phase 1 — 基础设施（已完成）
- `document_store.py`：DocumentStore 类（Chroma + bm25s + RRF + SQLite 元数据）
- `document_parser.py`：PDF/DOCX/HTML/TXT/MD 统一解析
- `persistence.py`：新增 collections、documents 表及 CRUD
- `server.py`：11 个文档库 API endpoints
- `tests/test_document_store.py`：核心操作测试

### Phase 2 — Pipeline 集成（已完成）
- `models.py`：`ResearchRequest` / `ResearchState` 新增 `document_collections`
- `subagent.py`：`_search_document_collections()` + 并行检索 + 文档来源高权重
- `synthesis.py`：`file://` citation 识别 + `Path.exists()` 验证
- `graph.py`：透传 `document_collections`，合并 sources
- `helpers.py`：`normalize_search_item` 保留原始 `source` 字段

### Phase 3 — 前端（已完成）
- `index.html`：文档库勾选面板
- `library.js` / `library.css`：文档库管理页面
- `api.js`：文档库 API 封装
- `app.js`：`/library` 路由
- `sources.js`：来源类型图标区分

### Phase 4 — 优化（已完成）
- 异步文档解析（pending → indexing → indexed/failed）
- 重新索引功能（单文档 + 全库）
- Category metadata 过滤
- 性能基准脚本（`scripts/benchmark_retrieval.py`）

### Bug 修复记录

| # | Bug | 根因 | 修复文件 |
|---|-----|------|----------|
| 1 | `python-multipart` 缺失 | FastAPI UploadFile 需要 | `pyproject.toml` |
| 2 | Chroma 维度不匹配 (512 vs 384) | 默认 embedding 函数与 bge-small-zh 维度不同 | `document_store.py` |
| 3 | `test_agents.py` mock 签名不匹配 | `fake_run` 未接收 `document_collections` | `tests/test_agents.py` |
| 4 | 上传后 doc_count 始终为 0 | SQLite `c.*` 同名列 `doc_count` 被默认值 0 覆盖 | `persistence.py` |
| 5 | 报告没有引用 RAG 文档 | `normalize_search_item` 覆盖 `source: "document"` 为 `"search"` | `helpers.py` |
| 6 | 报告卡在 adding citation | `synthesis.py` 缺少 `from pathlib import Path` | `synthesis.py` |
| 7 | 报告重复 References/Sources | subagent prompt 要求生成 Sources，synthesis 又追加 References | `prompts.py`, `synthesis.py` |
| 8 | 小文档库 BM25 warning | `k=24` 超过 corpus size 17 | `document_store.py` |

---

## 五、关键决策记录

| # | 决策项 | 结论 | 理由 |
|---|--------|------|------|
| 1 | Embedding 模型 | **BAAI/bge-small-zh-v1.5** | 384维同尺寸，中文最优，速度/大小/效果平衡 |
| 2 | 文档引用格式 | **`file://{abs_path}`** | 真实路径可直接用 `Path.exists()` 验证 |
| 3 | 混合检索 | **一期即实现**：Chroma 向量 + bm25s 关键词 + RRF | 纯向量检索在专业术语场景召回率低 |
| 4 | 前端框架 | **继续 Vanilla JS** | 如无必要勿增实体 |
| 5 | 中文分词 | **jieba** | 纯 Python，bm25s 兼容 |
| 6 | 异步解析 | **后台 asyncio.Task** | API 立即返回 pending，不阻塞用户体验 |
| 7 | 重复引用处理 | **Prompt 禁止 LLM 生成 Sources + 代码清理已有小节** | 统一由 `add_citations` 管理 References |

---

## 六、附录：目录结构

```
src/backend/
  server.py            (+ 文档库 API endpoints)
  models.py            (+ document_collections 字段)
  graph.py             (+ document_sources 合并逻辑)
  subagent.py          (+ 文档库混合检索逻辑)
  synthesis.py         (+ file:// citation + 重复引用清理)
  helpers.py           (+ 保留原始 source 字段)
  document_store.py    (NEW — Chroma + bm25s + RRF)
  document_parser.py   (NEW — PDF/DOCX/TXT/HTML 解析)
  persistence.py       (+ collections, documents 表)
  prompts.py           (- 移除自动 Sources/References 生成)

src/renderer/
  index.html           (+ 文档库勾选面板)
  css/style.css        (+ library 页面样式)
  js/api.js            (+ 文档库 API)
  js/library.js        (NEW — 文档库页面逻辑)
  js/app.js            (+ /library 路由)
  js/sources.js        (+ 来源类型图标)

scripts/
  benchmark_retrieval.py (NEW — 向量 vs BM25 vs Hybrid 基准)

~/.deep-research/
  history.db           (SQLite — runs, checkpoints, sources, collections, documents)
  chroma/              (Chroma 持久化数据)
  bm25/                (bm25s 序列化索引)
  docs/                (原始文件存储)
```
