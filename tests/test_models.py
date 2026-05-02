"""Tests for data models."""

import pytest


def test_subtask_defaults():
    from src.backend.models import Subtask
    st = Subtask(id="t1", title="Test Task")
    assert st.id == "t1"
    assert st.title == "Test Task"
    assert st.description == ""
    assert st.objective == ""
    assert st.output_format == "markdown"
    assert st.tool_guidance == "web search"
    assert st.source_types == "academic, official, news"
    assert st.boundaries == ""


def test_subtask_with_all_fields():
    from src.backend.models import Subtask
    st = Subtask(
        id="t1",
        title="Test",
        description="A description",
        objective="An objective",
        output_format="text",
        tool_guidance="code search",
        source_types="github",
        boundaries="Exclude X",
    )
    assert st.description == "A description"
    assert st.boundaries == "Exclude X"


def test_subtask_list():
    from src.backend.models import SubtaskList, Subtask
    sl = SubtaskList(subtasks=[Subtask(id="t1", title="T1"), Subtask(id="t2", title="T2")])
    assert len(sl.subtasks) == 2
    assert sl.subtasks[0].id == "t1"


def test_scaling_plan():
    from src.backend.models import ScalingPlan
    sp = ScalingPlan(complexity="moderate", subagent_count=5, tool_calls_per_subagent=15, target_sources=25)
    assert sp.complexity == "moderate"
    assert sp.subagent_count == 5
    assert sp.tool_calls_per_subagent == 15
    assert sp.target_sources == 25


def test_research_request_defaults():
    from src.backend.models import ResearchRequest
    req = ResearchRequest(query="test")
    assert req.query == "test"
    assert req.max_iterations is None
    assert req.quality_threshold is None
    assert req.provider is None
    assert req.model is None


def test_research_request_all_fields():
    from src.backend.models import ResearchRequest
    req = ResearchRequest(
        query="test",
        max_iterations=5,
        quality_threshold=0.8,
        provider="openai",
        model="gpt-4o",
    )
    assert req.max_iterations == 5
    assert req.quality_threshold == 0.8
    assert req.provider == "openai"
    assert req.model == "gpt-4o"


def test_research_response():
    from src.backend.models import ResearchResponse
    resp = ResearchResponse(run_id="abc123", status="started")
    assert resp.run_id == "abc123"
    assert resp.status == "started"


def test_config_update_request_all_none():
    from src.backend.models import ConfigUpdateRequest
    req = ConfigUpdateRequest()
    assert req.default_provider is None
    assert req.default_model is None
    assert req.max_iterations is None
    assert req.quality_threshold is None
    assert req.roles is None


def test_config_update_request_partial():
    from src.backend.models import ConfigUpdateRequest
    req = ConfigUpdateRequest(default_provider="openai", max_iterations=7)
    assert req.default_provider == "openai"
    assert req.max_iterations == 7
    assert req.default_model is None


def test_research_state_basic():
    from src.backend.models import ResearchState
    state: ResearchState = {}
    state["run_id"] = "test-run"
    state["user_query"] = "test query"
    state["iteration_count"] = 0
    state["events"] = []
    state["errors"] = []
    state["subagent_reports"] = []
    state["sources"] = []
    state["completed_subtasks"] = []
    state["quality_threshold"] = 0.7
    state["current_quality_score"] = 0.0
    state["max_iterations"] = 3
    state["research_complete"] = False
    state["memory"] = {}

    assert state["run_id"] == "test-run"
    assert state["user_query"] == "test query"
    assert state["quality_threshold"] == 0.7
