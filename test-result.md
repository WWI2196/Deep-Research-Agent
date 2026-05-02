# Test Results Archive

> 测试日期：2026-05-01
> 分支：reconstruct-v1
> 测试框架：pytest 9.x + pytest-asyncio

---

## 最终结果：177 passed, 0 failed

```
============================== 177 passed, 2 warnings in 21.10s ==============================
```

---

## 联调测试结果

| 端点 | 方法 | 状态 | 说明 |
|------|------|------|------|
| `/api/health` | GET | ✅ | 返回 provider/model/version |
| `/api/models` | GET | ✅ | 返回可用的 provider 列表 |
| `/api/config` | GET | ✅ | 返回完整配置（含 roles） |
| `/api/config` | POST | ✅ | 保存配置 |
| `/api/research` | POST | ✅ | 返回 run_id + status |
| `/api/research/stream` | POST | ✅ | **完整 8 阶段 pipeline 通过** |
| `/api/research/{id}/cancel` | POST | ✅ | 取消运行 |
| `/api/research/history` | GET | ✅ | 返回历史记录 |
| `/api/research/{id}/report` | GET | ✅ | 返回 run 报告 |

### SSE Pipeline 联调验证

`POST /api/research/stream` with `"query": "1+1=?"` 完整流程：

```
init → plan → split(5 subtasks) → scale(complex) → subagents(5 parallel)
  → subagent-complete x5 → reflection(max-iterations) → synthesize(16239 chars)
  → cite(40 sources → 16678 chars) → final-result → complete ✅
```

---

## 已修复的 Bug（共 4 个）

### BUG-1: `_chat()` 引用未定义变量
- **位置**: `agents.py:187`
- **修复**: `model=cfg.model` → `model=role_cfg.model`

### BUG-2: `build_and_run_graph()` 缺少 `nonlocal` 声明
- **位置**: `graph.py` — `_plan_node` 等 7 个节点函数
- **修复**: 各添加 `nonlocal completed_weight`

### BUG-3: `AppConfig` 缺少 `to_dict()` 方法
- **位置**: `config.py` — `AppConfig` 类
- **现象**: `GET /api/config` 返回 500
- **修复**: 添加 `to_dict()` 方法，返回 provider/model/roles 等字段

### BUG-4: `_emit()` 未 await async callback
- **位置**: `graph.py:41-43`
- **现象**: SSE 流只发出第一个事件，后续事件 `RuntimeWarning: coroutine was never awaited`
- **修复**: `_emit` 改为 `async def`，`on_event(evt)` 改为 `await on_event(evt)`，所有 `_emit()` 调用改为 `await _emit()`

---

## 测试覆盖详情

| 测试文件 | 用例数 | 通过 | 覆盖模块 |
|----------|--------|------|----------|
| test_config.py | 21 | 21 | 配置加载、优先级链、env 替换、to_dict round-trip、缓存 |
| test_search.py | 24 | 24 | Firecrawl 搜索/提取、DDG 退避、retry、标准化 |
| test_providers/test_providers.py | 10 | 10 | 工厂函数、retry 逻辑、致命/瞬时错误 |
| test_providers/test_anthropic.py | 6 | 6 | Anthropic chat、system 分离、多 text block |
| test_providers/test_openai_compatible.py | 4 | 4 | OpenAI chat、max_tokens、空 content |
| test_agents.py | 60 | 60 | 8 个 agent 函数 + 全部辅助函数 |
| test_models.py | 10 | 10 | Pydantic/TypedDict、默认值 |
| test_persistence.py | 9 | 9 | SQLite CRUD、checkpoint、schema |
| test_export.py | 3 | 3 | Markdown 导出 |
| test_graph.py | 10 | 10 | Pipeline、reflection loop、fallback、checkpoint |
| test_server.py | 17 | 17 | 全部 REST 端点、SSE streaming、CORS |
| test_integration.py | 3 | 3 | Config round-trip、persistence flow、search→agent |
| **合计** | **177** | **177** | |

---

## 遗留事项

- **功能限制**: `_extract_json` 不支持 JSON 数组（低优，prompt 不触发）
- **DeprecationWarning**: `@app.on_event("startup")` → 建议迁移为 lifespan handler
- **前端测试**: 需要 Vitest + jsdom，当前未配置
