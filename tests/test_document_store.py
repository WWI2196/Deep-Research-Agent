"""Tests for document parser, chunking, and store CRUD."""

import tempfile
from pathlib import Path

import pytest

from src.backend.document_parser import parse_document
from src.backend.document_store import (
    DocumentStore,
    _rrf_fusion,
    _split_text,
    _tokenize_chinese,
)


# ── Pure function tests ─────────────────────────────────────────

def test_split_text_basic():
    text = "Hello world. " * 100
    chunks = _split_text(text, chunk_size=50, overlap=5)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c) <= 50


def test_split_text_short():
    text = "Short text."
    chunks = _split_text(text)
    assert chunks == [text]


def test_tokenize_chinese():
    tokens = _tokenize_chinese("自然语言处理")
    assert len(tokens) >= 2  # jieba segments: ["自然语言", "处理"]


def test_rrf_fusion_basic():
    vector = [("a", 1), ("b", 2), ("c", 3)]
    bm25 = [("b", 1), ("d", 2)]
    scores = _rrf_fusion(vector, bm25)
    # b appears in both, should have highest score
    assert scores["b"] > scores["a"]
    assert scores["b"] > scores["d"]
    assert "c" in scores


# ── Parser tests ────────────────────────────────────────────────

def test_parse_text_file(tmp_path):
    path = tmp_path / "test.txt"
    path.write_text("Hello world\n\nSecond paragraph.", encoding="utf-8")
    result = parse_document(path)
    assert result["success"] is True
    assert "Hello world" in result["text"]
    assert result["page_count"] >= 1


def test_parse_markdown_file(tmp_path):
    path = tmp_path / "test.md"
    path.write_text("# Title\n\nBody text here.", encoding="utf-8")
    result = parse_document(path)
    assert result["success"] is True
    assert result["title"] == "Title"
    assert "Body text" in result["text"]


# ── DocumentStore CRUD tests ────────────────────────────────────

@pytest.fixture(autouse=True)
def temp_db_and_store_dirs(tmp_path):
    """Redirect persistence and document store to temp paths."""
    import src.backend.persistence as pmod
    from src.backend import document_store as dmod

    original_db_dir = pmod.DB_DIR
    original_db_path = pmod.DB_PATH
    original_chroma = dmod.CHROMA_DIR
    original_bm25 = dmod.BM25_DIR
    original_docs = dmod.DOCS_DIR

    pmod.DB_DIR = tmp_path
    pmod.DB_PATH = tmp_path / "history.db"
    dmod.CHROMA_DIR = tmp_path / "chroma"
    dmod.BM25_DIR = tmp_path / "bm25"
    dmod.DOCS_DIR = tmp_path / "docs"

    pmod.init_db()
    yield tmp_path

    pmod.DB_DIR = original_db_dir
    pmod.DB_PATH = original_db_path
    dmod.CHROMA_DIR = original_chroma
    dmod.BM25_DIR = original_bm25
    dmod.DOCS_DIR = original_docs


@pytest.mark.asyncio
async def test_create_list_delete_collection(tmp_path):
    store = DocumentStore(base_dir=tmp_path)
    col = await store.create_collection(name="Test Collection", description="desc")
    assert "id" in col
    assert col["name"] == "Test Collection"

    collections = await store.list_collections()
    assert len(collections) == 1
    assert collections[0]["name"] == "Test Collection"

    success = await store.delete_collection(col["id"])
    assert success is True

    collections = await store.list_collections()
    assert len(collections) == 0


@pytest.mark.asyncio
async def test_add_document_and_query(tmp_path):
    import asyncio

    store = DocumentStore(base_dir=tmp_path)
    col = await store.create_collection(name="Query Test")

    # Write a text file with specific content
    doc_path = tmp_path / "sample.txt"
    doc_path.write_text(
        "人工智能（AI）是计算机科学的一个分支，致力于创造能够执行通常需要人类智能的任务的机器。"
        "机器学习是 AI 的一个子领域，它使用统计技术让计算机系统能够从数据中学习和改进。"
        "深度学习是机器学习的一种方法，基于人工神经网络。",
        encoding="utf-8",
    )

    result = await store.add_document(col["id"], doc_path, name="sample.txt")
    assert result["success"] is True
    assert result["status"] == "pending"

    # Wait for background indexing to complete
    if store._pending_tasks:
        await asyncio.gather(*store._pending_tasks, return_exceptions=True)

    # Verify indexed
    docs = await store.list_documents(col["id"])
    assert len(docs) == 1
    assert docs[0]["status"] == "indexed"
    assert docs[0]["chunk_count"] > 0

    # Hybrid query
    results = await store.query([col["id"]], "什么是人工智能", top_k=3)
    assert len(results) > 0
    # Top result should mention AI
    assert "人工智能" in results[0]["text"] or "AI" in results[0]["text"]

    # Delete document
    doc_id = docs[0]["id"]
    success = await store.delete_document(col["id"], doc_id)
    assert success is True

    docs_after = await store.list_documents(col["id"])
    assert len(docs_after) == 0
