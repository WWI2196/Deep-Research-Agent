"""Tests for query_similarity, generate_broader_queries helpers."""

import pytest
from src.backend.helpers import query_similarity, generate_broader_queries


def test_query_similarity_identical():
    assert query_similarity("ai safety", "ai safety") == 1.0


def test_query_similarity_overlapping():
    assert query_similarity("ai safety", "ai safety regulations") == 2 / 3


def test_query_similarity_disjoint():
    assert query_similarity("ai safety", "quantum computing") == 0.0


def test_query_similarity_empty():
    assert query_similarity("", "test") == 0.0


def test_generate_broader_queries_strips_modifier():
    result = generate_broader_queries("AI safety research paper")
    assert len(result) >= 1
    assert all("research paper" not in q.lower() for q in result)


def test_generate_broader_queries_no_modifier():
    result = generate_broader_queries("AI safety")
    assert isinstance(result, list)
    assert len(result) <= 2


def test_generate_broader_queries_modifier_in_middle():
    result = generate_broader_queries("study of AI safety")
    assert len(result) >= 1
    assert all("study" not in q.lower().split() for q in result)


def test_generate_broader_queries_fallback_first_word_drop():
    result = generate_broader_queries("one two three four")
    assert len(result) >= 1
    # One fallback should drop "one"
    assert any("two three four" == r for r in result)
