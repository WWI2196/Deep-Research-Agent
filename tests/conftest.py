"""Shared test fixtures."""

import pytest
from unittest.mock import AsyncMock, MagicMock


@pytest.fixture
def mock_provider():
    """Return a mock LLM provider that returns canned responses."""
    provider = AsyncMock()
    provider.name = "mock"
    provider.chat.return_value = '{"subtasks": [{"id": "test", "title": "Test", "description": "Test task"}]}'
    return provider


@pytest.fixture
def sample_subtasks():
    return [
        {
            "id": "task_1",
            "title": "Market Analysis",
            "description": "Analyze market trends",
            "objective": "Understand market size and growth",
            "output_format": "markdown",
            "tool_guidance": "web search",
            "source_types": "news, official",
            "boundaries": "Exclude technical details",
        },
        {
            "id": "task_2",
            "title": "Technology Review",
            "description": "Review key technologies",
            "objective": "Identify emerging technologies",
            "output_format": "markdown",
            "tool_guidance": "web search",
            "source_types": "academic, official",
            "boundaries": "Exclude business aspects",
        },
    ]


@pytest.fixture
def sample_search_results():
    return {
        "data": [
            {"title": "Test Result 1", "url": "https://example.com/1", "description": "A test result"},
            {"title": "Test Result 2", "url": "https://example.com/2", "description": "Another test result"},
        ]
    }
