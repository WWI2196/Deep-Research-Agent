"""Tests for ReAct agent loop (react_agent.py)."""

import json
import pytest
from unittest.mock import AsyncMock


class FakeTool:
    def __init__(self, name, result):
        self.name = name
        self.description = f"Fake {name} tool"
        self.params_schema = {}
        self.result = result

    async def execute(self, **kwargs):
        return self.result


# ── run_react_agent ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_react_agent_executes_tool_and_returns_final_answer():
    from src.backend.react_agent import run_react_agent

    calls = []

    async def fake_chat(role, messages, temperature=0.3):
        calls.append(len(calls))
        step = len(calls)
        if step == 1:
            return json.dumps({
                "thought": "I need to search",
                "action": "search",
                "action_input": {"query": "test"},
            })
        return json.dumps({
            "thought": "I have enough info",
            "final_answer": "# Report\n\nAnswer here.",
        })

    tool = FakeTool("search", {"success": True, "result": {"results": [{"url": "https://x.com", "title": "X"}]}})
    result = await run_react_agent(
        system_prompt="You are a test agent",
        user_prompt="Do research",
        tools=[tool],
        chat_fn=fake_chat,
        max_steps=5,
    )

    assert "# Report" in result["final_answer"]
    assert result["steps_taken"] == 2
    assert len(result["tool_calls"]) == 1
    assert result["tool_calls"][0]["tool"] == "search"


@pytest.mark.asyncio
async def test_react_agent_respects_max_steps():
    from src.backend.react_agent import run_react_agent

    async def fake_chat(role, messages, temperature=0.3):
        return json.dumps({
            "thought": "Keep going",
            "action": "search",
            "action_input": {"query": "test"},
        })

    tool = FakeTool("search", {"success": True, "result": {"results": []}})
    result = await run_react_agent(
        system_prompt="sys",
        user_prompt="usr",
        tools=[tool],
        chat_fn=fake_chat,
        max_steps=3,
    )

    assert result["steps_taken"] == 3
    assert result["error"] is not None
    assert "max_steps" in result["error"]


@pytest.mark.asyncio
async def test_react_agent_handles_tool_error():
    from src.backend.react_agent import run_react_agent

    calls = []

    async def fake_chat(role, messages, temperature=0.3):
        calls.append(len(calls))
        if len(calls) == 1:
            return json.dumps({
                "thought": "Search now",
                "action": "search",
                "action_input": {"query": "test"},
            })
        return json.dumps({
            "thought": "Search failed, I'll answer with what I know",
            "final_answer": "# Fallback\n\nBest effort answer.",
        })

    tool = FakeTool("search", {"success": False, "error": "Network down"})
    result = await run_react_agent(
        system_prompt="sys",
        user_prompt="usr",
        tools=[tool],
        chat_fn=fake_chat,
        max_steps=5,
    )

    assert "Fallback" in result["final_answer"]
    assert len(result["tool_calls"]) == 1
    assert result["tool_calls"][0]["result"]["success"] is False


@pytest.mark.asyncio
async def test_react_agent_unknown_tool_retries():
    from src.backend.react_agent import run_react_agent

    calls = []

    async def fake_chat(role, messages, temperature=0.3):
        calls.append(len(calls))
        if len(calls) == 1:
            return json.dumps({
                "thought": "Wrong tool",
                "action": "nonexistent",
                "action_input": {},
            })
        return json.dumps({
            "thought": "I'll answer directly",
            "final_answer": "# Direct\n\nAnswer.",
        })

    tool = FakeTool("search", {"success": True, "result": {}})
    result = await run_react_agent(
        system_prompt="sys",
        user_prompt="usr",
        tools=[tool],
        chat_fn=fake_chat,
        max_steps=5,
    )

    assert "Direct" in result["final_answer"]


@pytest.mark.asyncio
async def test_react_agent_parses_markdown_wrapped_json():
    from src.backend.react_agent import run_react_agent

    async def fake_chat(role, messages, temperature=0.3):
        return '```json\n{"thought": "Done", "final_answer": "# OK"}\n```'

    result = await run_react_agent(
        system_prompt="sys",
        user_prompt="usr",
        tools=[],
        chat_fn=fake_chat,
        max_steps=5,
    )

    assert result["final_answer"] == "# OK"


# ── _parse_react_output ─────────────────────────────────────────

def test_parse_react_output_variants():
    from src.backend.react_agent import _parse_react_output

    assert _parse_react_output('{"action": "x"}')["action"] == "x"
    assert _parse_react_output('```json\n{"action": "x"}\n```')["action"] == "x"
    assert _parse_react_output('```\n{"action": "x"}\n```')["action"] == "x"
    assert _parse_react_output('some text {"action": "x"} more')["action"] == "x"
    assert _parse_react_output("not json") is None
    assert _parse_react_output("") is None


# ── Context compression ─────────────────────────────────────────

def test_should_compress_messages_threshold():
    from src.backend.react_agent import _should_compress_messages, COMPRESSION_THRESHOLD_CHARS

    # Below threshold
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "usr"},
        {"role": "assistant", "content": "a"},
        {"role": "user", "content": "b"},
    ]
    assert _should_compress_messages(messages) is False

    # Above threshold
    long_content = "x" * (COMPRESSION_THRESHOLD_CHARS + 1)
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "usr"},
        {"role": "assistant", "content": long_content},
    ]
    assert _should_compress_messages(messages) is True


@pytest.mark.asyncio
async def test_compress_messages_keeps_recent_rounds():
    from src.backend.react_agent import _compress_messages

    async def fake_chat(role, messages, temperature=0.3):
        return "Summary of old work."

    # Build 14 messages: [system, user] + 6 rounds (assistant+user)
    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "init"},
    ]
    for i in range(6):
        msgs.append({"role": "assistant", "content": f"action {i}"})
        msgs.append({"role": "user", "content": f"observation {i}"})

    result = await _compress_messages(msgs, fake_chat, 0.3, keep_recent=5)

    # Head preserved
    assert result[0]["role"] == "system"
    assert result[1]["role"] == "user"
    assert result[1]["content"] == "init"

    # Summary inserted
    assert result[2]["role"] == "user"
    assert "Summary of old work" in result[2]["content"]

    # Tail preserved: last 5 rounds = 10 messages
    assert len(result) == 3 + 10
    assert result[3]["content"] == "action 1"
    assert result[-2]["content"] == "action 5"
    assert result[-1]["content"] == "observation 5"


@pytest.mark.asyncio
async def test_compress_messages_fallback_on_llm_failure():
    from src.backend.react_agent import _compress_messages

    async def failing_chat(role, messages, temperature=0.3):
        raise RuntimeError("LLM down")

    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "init"},
        {"role": "assistant", "content": "old action"},
        {"role": "user", "content": "old obs"},
        {"role": "assistant", "content": "recent action"},
        {"role": "user", "content": "recent obs"},
    ]

    result = await _compress_messages(msgs, failing_chat, 0.3, keep_recent=1)

    # Should still preserve head + tail and insert fallback notice
    assert result[0]["content"] == "sys"
    assert result[1]["content"] == "init"
    assert "truncated" in result[2]["content"].lower()
    assert result[3]["content"] == "recent action"
    assert result[4]["content"] == "recent obs"


def test_compress_messages_noop_when_too_few():
    from src.backend.react_agent import _compress_messages

    # _compress_messages is async, but we can test the sync guard via asyncio.run
    import asyncio

    async def fake_chat(role, messages, temperature=0.3):
        return "summary"

    msgs = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "init"},
        {"role": "assistant", "content": "a"},
        {"role": "user", "content": "b"},
    ]

    result = asyncio.run(_compress_messages(msgs, fake_chat, 0.3, keep_recent=5))
    assert result == msgs
