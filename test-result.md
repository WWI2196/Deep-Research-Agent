# Test Results

> 日期：2026-05-04
> 分支：main
> 测试框架：pytest 9.x + pytest-asyncio

---

## 最终结果：160 passed, 0 failed

```
============================== 160 passed, 2 warnings in 12.74s =======================
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
| `/api/research/stream` | POST | ✅ | 完整 8 阶段 pipeline 通过 |
| `/api/research/{id}/cancel` | POST | ✅ | 取消运行 |
| `/api/research/history` | GET | ✅ | 返回历史记录 |
| `/api/research/{id}/report` | GET | ✅ | 返回 run 报告 |

### SSE Pipeline 验证

`POST /api/research/stream` with query 完整流程：

```
init → plan → split(N subtasks) → scale → subagents(N parallel)
  → subagent-complete xN → reflection → synthesize → cite → final-result → complete ✅
```

---

## 测试覆盖详情

| 测试文件 | 用例数 | 覆盖模块 |
|----------|--------|----------|
| test_config.py | 21 | 配置加载、优先级链、env 替换、to_dict、缓存 |
| test_search.py | 7 | SearXNG 搜索、trafilatura 提取、fallback |
| test_providers/test_providers.py | 10 | 工厂函数、retry 逻辑、致命/瞬时错误 |
| test_providers/test_anthropic.py | 6 | Anthropic chat、system 分离、多 text block |
| test_providers/test_openai_compatible.py | 4 | OpenAI chat、max_tokens、空 content |
| test_agents.py | 60 | 8 个 agent 函数 + 辅助函数 |
| test_models.py | 10 | Pydantic/TypedDict、默认值 |
| test_persistence.py | 9 | SQLite CRUD、checkpoint、schema |
| test_export.py | 3 | Markdown 导出 |
| test_graph.py | 10 | Pipeline、reflection loop、fallback、checkpoint |
| test_server.py | 17 | 全部 REST 端点、SSE streaming、CORS |
| test_integration.py | 3 | Config round-trip、persistence flow、search→agent |
| **合计** | **160** | |

---

## 遗留事项

- **Warning**: `@app.on_event("startup")` 已废弃，建议迁移为 lifespan handler
- **功能限制**: `_extract_json` 不支持 JSON 数组（低优，prompt 不触发）
- **前端测试**: 需要 Vitest + jsdom，当前未配置
