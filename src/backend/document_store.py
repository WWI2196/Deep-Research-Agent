"""Document store — Chroma vector + bm25s keyword + RRF hybrid retrieval.

All public methods are async; blocking IO runs via asyncio.to_thread.
"""

import asyncio
import json
import logging
import uuid
from pathlib import Path
from typing import Any

import bm25s
import chromadb
import jieba

from .document_parser import parse_document
from .persistence import (
    persist_collection,
    persist_document,
    delete_collection_db,
    delete_document_db,
    list_collections_db,
    list_documents_db,
    get_collection_db,
    update_document_status,
    get_document_db,
)
from .tracing import trace

logger = logging.getLogger(__name__)

# ── chunking defaults ───────────────────────────────────────────

_CHUNK_SIZE = 800
_CHUNK_OVERLAP = 100
_SEPARATORS = ["\n\n", "\n", ". ", "。", "！", "？", " ", ""]

DB_DIR = Path.home() / ".deep-research"
CHROMA_DIR = DB_DIR / "chroma"
BM25_DIR = DB_DIR / "bm25"
DOCS_DIR = DB_DIR / "docs"


# ── helpers ─────────────────────────────────────────────────────

def _split_text(text: str, chunk_size: int = _CHUNK_SIZE, overlap: int = _CHUNK_OVERLAP) -> list[str]:
    """Recursive text splitter with semantic separators."""
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    chunks: list[str] = []
    for sep in _SEPARATORS:
        if sep in text:
            parts = text.split(sep)
            current = ""
            for part in parts:
                candidate = (current + sep + part).strip() if current else part.strip()
                if len(candidate) <= chunk_size:
                    current = candidate
                else:
                    if current:
                        chunks.append(current)
                    current = part.strip()
            if current:
                chunks.append(current)
            break
    else:
        # Hard split by characters
        for i in range(0, len(text), chunk_size - overlap):
            chunk = text[i:i + chunk_size].strip()
            if chunk:
                chunks.append(chunk)

    # Recursively split oversized chunks
    result: list[str] = []
    for chunk in chunks:
        if len(chunk) > chunk_size:
            result.extend(_split_text(chunk, chunk_size, overlap))
        else:
            result.append(chunk)
    return result


def _tokenize_chinese(text: str) -> list[str]:
    return list(jieba.cut(text))


def _rrf_fusion(
    vector_results: list[tuple[str, int]],
    bm25_results: list[tuple[str, int]],
    k: int = 60,
) -> dict[str, float]:
    """Reciprocal Rank Fusion. Inputs: [(chunk_id, rank), ...]. Returns {chunk_id: score}."""
    scores: dict[str, float] = {}
    for chunk_id, rank in vector_results:
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    for chunk_id, rank in bm25_results:
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return scores


# ── DocumentStore ───────────────────────────────────────────────

class DocumentStore:
    """Manages collections of documents with hybrid (vector + BM25) retrieval."""

    def __init__(self, base_dir: Path | None = None) -> None:
        self._base_dir = base_dir or DB_DIR
        self._chroma_dir = self._base_dir / "chroma"
        self._bm25_dir = self._base_dir / "bm25"
        self._docs_dir = self._base_dir / "docs"

        for d in (self._chroma_dir, self._bm25_dir, self._docs_dir):
            d.mkdir(parents=True, exist_ok=True)

        self._chroma_client = chromadb.PersistentClient(path=str(self._chroma_dir))
        self._embedding_model: Any | None = None
        self._model_lock = asyncio.Lock()
        self._indexing_locks: dict[str, asyncio.Lock] = {}
        self._pending_tasks: set[asyncio.Task] = set()

    # ── indexing lock per collection ────────────────────────────

    def _get_indexing_lock(self, collection_id: str) -> asyncio.Lock:
        if collection_id not in self._indexing_locks:
            self._indexing_locks[collection_id] = asyncio.Lock()
        return self._indexing_locks[collection_id]

    # ── embedding lazy init ─────────────────────────────────────

    async def _get_embedding_model(self) -> Any:
        if self._embedding_model is not None:
            return self._embedding_model
        async with self._model_lock:
            if self._embedding_model is not None:
                return self._embedding_model
            from sentence_transformers import SentenceTransformer
            self._embedding_model = await asyncio.to_thread(
                SentenceTransformer, "BAAI/bge-small-zh-v1.5"
            )
            logger.info("Loaded embedding model: BAAI/bge-small-zh-v1.5")
        return self._embedding_model

    # ── collection CRUD ─────────────────────────────────────────

    async def create_collection(self, name: str, description: str = "") -> dict[str, Any]:
        collection_id = str(uuid.uuid4())
        await asyncio.to_thread(
            self._chroma_client.create_collection,
            name=collection_id,
            metadata={"name": name, "description": description},
        )
        await persist_collection(collection_id, name, description)
        logger.info("Created collection %s (%s)", name, collection_id)
        return {"id": collection_id, "name": name, "description": description, "doc_count": 0}

    async def delete_collection(self, collection_id: str) -> bool:
        try:
            await asyncio.to_thread(
                self._chroma_client.delete_collection, name=collection_id
            )
        except Exception as exc:
            logger.warning("Chroma delete_collection %s: %s", collection_id, exc)

        # Remove bm25 index
        bm25_path = self._bm25_dir / collection_id
        if bm25_path.exists():
            import shutil
            await asyncio.to_thread(shutil.rmtree, bm25_path, ignore_errors=True)

        # Remove doc files
        doc_dir = self._docs_dir / collection_id
        if doc_dir.exists():
            import shutil
            await asyncio.to_thread(shutil.rmtree, doc_dir, ignore_errors=True)

        await delete_collection_db(collection_id)
        logger.info("Deleted collection %s", collection_id)
        return True

    async def list_collections(self) -> list[dict[str, Any]]:
        rows = await list_collections_db()
        return [
            {
                "id": r["id"],
                "name": r["name"],
                "description": r.get("description", ""),
                "doc_count": r.get("doc_count", 0),
                "created_at": r.get("created_at", 0),
            }
            for r in rows
        ]

    async def get_collection(self, collection_id: str) -> dict[str, Any] | None:
        row = await get_collection_db(collection_id)
        if not row:
            return None
        return {
            "id": row["id"],
            "name": row["name"],
            "description": row.get("description", ""),
            "doc_count": row.get("doc_count", 0),
            "created_at": row.get("created_at", 0),
        }

    # ── document CRUD ───────────────────────────────────────────

    async def add_document(
        self,
        collection_id: str,
        file_path: Path,
        name: str | None = None,
        category: str = "",
        tags: list[str] | None = None,
    ) -> dict[str, Any]:
        """Stage a document for async indexing. Returns immediately with pending status."""
        doc_id = str(uuid.uuid4())
        file_path = Path(file_path)
        ext = file_path.suffix.lower()
        doc_name = name or file_path.name
        tags = tags or []

        # Copy to storage
        dest_dir = self._docs_dir / collection_id
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / f"{doc_id}{ext}"
        await asyncio.to_thread(
            file_path.read_bytes
        )  # ensure readable
        import shutil
        await asyncio.to_thread(shutil.copy2, file_path, dest_path)

        # Persist as pending
        await persist_document(
            doc_id=doc_id,
            collection_id=collection_id,
            name=doc_name,
            file_path=str(dest_path),
            file_type=ext.lstrip("."),
            category=category,
            tags=json.dumps(tags),
            page_count=0,
            chunk_count=0,
            status="pending",
        )

        # Launch background indexing
        task = asyncio.create_task(
            self._process_document(collection_id, doc_id, doc_name, str(dest_path), ext, category, tags)
        )
        self._pending_tasks.add(task)
        task.add_done_callback(self._pending_tasks.discard)

        logger.info("Staged document %s (%s) for async indexing", doc_name, doc_id)
        return {
            "id": doc_id,
            "name": doc_name,
            "success": True,
            "status": "pending",
            "chunk_count": 0,
            "error": "",
        }

    async def _process_document(
        self,
        collection_id: str,
        doc_id: str,
        doc_name: str,
        dest_path: str,
        ext: str,
        category: str,
        tags: list[str],
    ) -> None:
        """Background task: parse, chunk, embed, and index a document."""
        async with self._get_indexing_lock(collection_id):
            await update_document_status(doc_id, status="indexing")

            try:
                parse_result = await asyncio.to_thread(parse_document, dest_path)
                if not parse_result["success"]:
                    await update_document_status(
                        doc_id, status="failed", error_msg=parse_result.get("error", "Parse failed")
                    )
                    return

                text = parse_result["text"]
                chunks = _split_text(text)

                if not chunks:
                    await update_document_status(
                        doc_id, status="failed", error_msg="No text chunks generated"
                    )
                    return

                # Embed and add to Chroma
                model = await self._get_embedding_model()
                embeddings = await asyncio.to_thread(model.encode, chunks)

                chunk_ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
                metadatas = [
                    {
                        "doc_id": doc_id,
                        "doc_name": doc_name,
                        "collection_id": collection_id,
                        "category": category,
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "source_type": ext.lstrip("."),
                        "file_path": dest_path,
                    }
                    for i in range(len(chunks))
                ]

                collection = self._chroma_client.get_or_create_collection(name=collection_id)
                await asyncio.to_thread(
                    collection.add,
                    ids=chunk_ids,
                    documents=chunks,
                    embeddings=embeddings.tolist(),
                    metadatas=metadatas,
                )

                # Rebuild bm25 index for the whole collection
                await self._rebuild_bm25_index(collection_id)

                # Update status
                await update_document_status(
                    doc_id,
                    status="indexed",
                    chunk_count=len(chunks),
                    page_count=parse_result.get("page_count", 0),
                )

                logger.info(
                    "Indexed document %s in collection %s (%d chunks)",
                    doc_name, collection_id, len(chunks),
                )
            except Exception as exc:
                logger.exception("Failed to index document %s: %s", doc_id, exc)
                await update_document_status(doc_id, status="failed", error_msg=str(exc))

    async def delete_document(self, collection_id: str, doc_id: str) -> bool:
        # Cancel any pending indexing task for this doc (best-effort)
        for task in list(self._pending_tasks):
            if task.get_name() == f"index-{doc_id}":
                task.cancel()

        # Remove from Chroma
        try:
            collection = self._chroma_client.get_collection(name=collection_id)
            all_data = await asyncio.to_thread(collection.get)
            ids_to_remove = [
                cid for cid in all_data.get("ids", [])
                if cid.startswith(f"{doc_id}_chunk_")
            ]
            if ids_to_remove:
                await asyncio.to_thread(collection.delete, ids=ids_to_remove)
        except Exception as exc:
            logger.warning("Chroma delete document %s: %s", doc_id, exc)

        # Delete file
        doc_path = _get_document_file_path(collection_id, doc_id)
        if doc_path and doc_path.exists():
            await asyncio.to_thread(doc_path.unlink, missing_ok=True)

        # Rebuild bm25
        await self._rebuild_bm25_index(collection_id)
        await delete_document_db(collection_id, doc_id)
        logger.info("Deleted document %s from collection %s", doc_id, collection_id)
        return True

    async def list_documents(self, collection_id: str) -> list[dict[str, Any]]:
        rows = await list_documents_db(collection_id)
        return [
            {
                "id": r["id"],
                "name": r["name"],
                "file_type": r.get("file_type", ""),
                "category": r.get("category", ""),
                "tags": json.loads(r["tags"]) if r.get("tags") else [],
                "page_count": r.get("page_count", 0),
                "chunk_count": r.get("chunk_count", 0),
                "status": r.get("status", ""),
                "error_msg": r.get("error_msg", ""),
                "created_at": r.get("created_at", 0),
            }
            for r in rows
        ]

    # ── re-indexing ─────────────────────────────────────────────

    async def reindex_document(self, collection_id: str, doc_id: str) -> dict[str, Any]:
        """Re-parse and re-index a single document."""
        row = await get_document_db(doc_id)
        if not row or row["collection_id"] != collection_id:
            return {"success": False, "error": "Document not found"}

        # Delete existing chunks from Chroma first
        try:
            collection = self._chroma_client.get_collection(name=collection_id)
            all_data = await asyncio.to_thread(collection.get)
            ids_to_remove = [
                cid for cid in all_data.get("ids", [])
                if cid.startswith(f"{doc_id}_chunk_")
            ]
            if ids_to_remove:
                await asyncio.to_thread(collection.delete, ids=ids_to_remove)
        except Exception as exc:
            logger.warning("Chroma cleanup before reindex %s: %s", doc_id, exc)

        # Reset status and re-process
        await update_document_status(doc_id, status="pending", chunk_count=0, error_msg="")
        task = asyncio.create_task(
            self._process_document(
                collection_id=collection_id,
                doc_id=doc_id,
                doc_name=row["name"],
                dest_path=row["file_path"],
                ext=f".{row.get('file_type', '')}" if row.get("file_type") else "",
                category=row.get("category", ""),
                tags=json.loads(row["tags"]) if row.get("tags") else [],
            ),
            name=f"index-{doc_id}",
        )
        self._pending_tasks.add(task)
        task.add_done_callback(self._pending_tasks.discard)

        return {"success": True, "id": doc_id, "status": "pending"}

    async def reindex_collection(self, collection_id: str) -> dict[str, Any]:
        """Re-index all documents in a collection."""
        docs = await list_documents_db(collection_id)
        if not docs:
            return {"success": True, "reindexed": 0}

        # Clear Chroma collection entirely
        try:
            await asyncio.to_thread(
                self._chroma_client.delete_collection, name=collection_id
            )
            await asyncio.to_thread(
                self._chroma_client.create_collection,
                name=collection_id,
                metadata={"name": collection_id},
            )
        except Exception as exc:
            logger.warning("Chroma reset for reindex %s: %s", collection_id, exc)

        reindexed = 0
        for row in docs:
            doc_id = row["id"]
            await update_document_status(doc_id, status="pending", chunk_count=0, error_msg="")
            task = asyncio.create_task(
                self._process_document(
                    collection_id=collection_id,
                    doc_id=doc_id,
                    doc_name=row["name"],
                    dest_path=row["file_path"],
                    ext=f".{row.get('file_type', '')}" if row.get("file_type") else "",
                    category=row.get("category", ""),
                    tags=json.loads(row["tags"]) if row.get("tags") else [],
                ),
                name=f"index-{doc_id}",
            )
            self._pending_tasks.add(task)
            task.add_done_callback(self._pending_tasks.discard)
            reindexed += 1

        # Remove old bm25 index; will be rebuilt per document
        bm25_path = self._bm25_dir / collection_id
        if bm25_path.exists():
            import shutil
            await asyncio.to_thread(shutil.rmtree, bm25_path, ignore_errors=True)

        logger.info("Reindexing collection %s (%d documents)", collection_id, reindexed)
        return {"success": True, "reindexed": reindexed}

    # ── bm25s index management ──────────────────────────────────

    async def _rebuild_bm25_index(self, collection_id: str) -> None:
        """Rebuild bm25s index from all chunks in a Chroma collection."""
        try:
            collection = self._chroma_client.get_collection(name=collection_id)
        except Exception:
            logger.warning("Collection %s not found for bm25 rebuild", collection_id)
            return

        all_data = await asyncio.to_thread(collection.get)
        ids = all_data.get("ids", [])
        documents = all_data.get("documents", [])

        if not documents:
            # No documents — remove index if exists
            bm25_path = self._bm25_dir / collection_id
            if bm25_path.exists():
                import shutil
                await asyncio.to_thread(shutil.rmtree, bm25_path, ignore_errors=True)
            return

        # Tokenize Chinese
        tokenized_corpus = bm25s.tokenize([
            " ".join(_tokenize_chinese(doc)) for doc in documents
        ])

        retriever = bm25s.BM25()
        await asyncio.to_thread(retriever.index, tokenized_corpus)

        # Save
        bm25_path = self._bm25_dir / collection_id
        bm25_path.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(retriever.save, str(bm25_path))

        # Save corpus mapping
        corpus_mapping = {
            id_: {"text": doc} for id_, doc in zip(ids, documents)
        }
        corpus_file = bm25_path / "corpus.json"
        await asyncio.to_thread(
            corpus_file.write_text,
            json.dumps(corpus_mapping, ensure_ascii=False),
            encoding="utf-8",
        )

        logger.info("Rebuilt bm25 index for collection %s (%d chunks)", collection_id, len(documents))

    # ── hybrid search ───────────────────────────────────────────

    async def query(
        self,
        collection_ids: list[str],
        query_text: str,
        top_k: int = 10,
        category_filter: str | None = None,
    ) -> list[dict[str, Any]]:
        """Hybrid search across collections. Returns top-k chunks."""
        if not collection_ids:
            return []

        all_results: list[dict[str, Any]] = []

        for cid in collection_ids:
            try:
                fused = await self._query_single_collection(cid, query_text, top_k, category_filter)
                all_results.extend(fused)
            except Exception as exc:
                logger.warning("Query collection %s failed: %s", cid, exc)

        # Sort by RRF score descending
        all_results.sort(key=lambda x: x["score"], reverse=True)
        return all_results[:top_k]

    async def _query_single_collection(
        self, collection_id: str, query_text: str, top_k: int, category_filter: str | None = None
    ) -> list[dict[str, Any]]:
        """Vector + BM25 + RRF for a single collection."""
        # ── vector search ──
        model = await self._get_embedding_model()
        query_embedding = await asyncio.to_thread(model.encode, [query_text])
        collection = self._chroma_client.get_or_create_collection(name=collection_id)

        vec_kwargs: dict[str, Any] = {
            "query_embeddings": query_embedding.tolist(),
            "n_results": top_k * 2,
            "include": ["documents", "metadatas", "distances"],
        }
        if category_filter:
            vec_kwargs["where"] = {"category": category_filter}

        vec_result = await asyncio.to_thread(collection.query, **vec_kwargs)

        vector_ranking: list[tuple[str, int]] = []
        vector_meta: dict[str, dict[str, Any]] = {}
        vector_text: dict[str, str] = {}
        ids_list = vec_result.get("ids", [[]])[0]
        docs_list = vec_result.get("documents", [[]])[0]
        meta_list = vec_result.get("metadatas", [[]])[0]
        for rank, (cid, text, meta) in enumerate(zip(ids_list, docs_list, meta_list), start=1):
            vector_ranking.append((cid, rank))
            vector_meta[cid] = meta
            vector_text[cid] = text

        await trace("subagents", "rag_vector_results", f"Vector search returned {len(vector_ranking)} chunks", {
            "collection_id": collection_id,
            "query": query_text,
            "n_results": top_k * 2,
            "category_filter": category_filter,
            "chunks": [
                {"chunk_id": cid, "rank": rank, "doc_name": vector_meta.get(cid, {}).get("doc_name", ""), "distance": vec_result.get("distances", [[]])[0][i] if vec_result.get("distances") and i < len(vec_result["distances"][0]) else None}
                for i, (cid, rank) in enumerate(vector_ranking[:10])
            ],
        }, level="debug")

        # ── bm25 search ──
        bm25_path = self._bm25_dir / collection_id
        bm25_ranking: list[tuple[str, int]] = []
        bm25_text: dict[str, str] = {}

        if bm25_path.exists() and (bm25_path / "corpus.json").exists():
            try:
                retriever = bm25s.BM25()
                retriever = await asyncio.to_thread(retriever.load, str(bm25_path))

                query_tokens = bm25s.tokenize(" ".join(_tokenize_chinese(query_text)))
                corpus_size = len(corpus_ids)
                bm25_k = min(top_k * 2, corpus_size)
                results, _scores = await asyncio.to_thread(
                    retriever.retrieve, query_tokens, k=bm25_k
                )

                # Load corpus mapping for text lookup
                corpus_file = bm25_path / "corpus.json"
                corpus_raw = await asyncio.to_thread(
                    corpus_file.read_text, encoding="utf-8"
                )
                corpus_map = json.loads(corpus_raw)

                # results shape: [[doc_idx_0, doc_idx_1, ...]]
                # We need to map back to Chroma IDs
                # corpus.json keys are Chroma chunk IDs
                corpus_ids = list(corpus_map.keys())
                for rank, idx in enumerate(results[0], start=1):
                    if idx < len(corpus_ids):
                        cid = corpus_ids[idx]
                        bm25_ranking.append((cid, rank))
                        bm25_text[cid] = corpus_map[cid].get("text", "")
            except Exception as exc:
                logger.warning("BM25 search failed for %s: %s", collection_id, exc)

        await trace("subagents", "rag_bm25_results", f"BM25 search returned {len(bm25_ranking)} chunks", {
            "collection_id": collection_id,
            "query": query_text,
            "corpus_size": len(corpus_ids) if 'corpus_ids' in locals() else 0,
            "bm25_k": min(top_k * 2, len(corpus_ids)) if 'corpus_ids' in locals() else top_k * 2,
            "chunks": [
                {"chunk_id": cid, "rank": rank}
                for cid, rank in bm25_ranking[:10]
            ],
        }, level="debug")

        # ── RRF fusion ──
        rrf_scores = _rrf_fusion(vector_ranking, bm25_ranking)
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

        await trace("subagents", "rag_rrf_fusion", f"RRF fusion combined {len(vector_ranking)} vector + {len(bm25_ranking)} bm25 into {len(rrf_scores)} unique chunks", {
            "collection_id": collection_id,
            "vector_count": len(vector_ranking),
            "bm25_count": len(bm25_ranking),
            "fusion_count": len(rrf_scores),
            "top_scores": [
                {"chunk_id": cid, "score": round(rrf_scores[cid], 4)}
                for cid in sorted_ids[:10]
            ],
        }, level="debug")

        # ── assemble results ──
        output: list[dict[str, Any]] = []
        for cid in sorted_ids[:top_k]:
            meta = vector_meta.get(cid, {})
            text = vector_text.get(cid) or bm25_text.get(cid, "")

            # Apply category filter post-RRF if BM25 results lack metadata
            if category_filter and meta.get("category") != category_filter:
                # Try to load metadata from Chroma for BM25-only hits
                if not meta:
                    try:
                        chroma_meta = await asyncio.to_thread(
                            collection.get, ids=[cid], include=["metadatas"]
                        )
                        meta_list = chroma_meta.get("metadatas", [[]])[0]
                        if meta_list:
                            meta = meta_list[0]
                        if meta.get("category") != category_filter:
                            continue
                    except Exception:
                        continue
                else:
                    continue

            output.append({
                "chunk_id": cid,
                "collection_id": collection_id,
                "doc_id": meta.get("doc_id", ""),
                "doc_name": meta.get("doc_name", ""),
                "text": text,
                "score": rrf_scores[cid],
                "category": meta.get("category", ""),
                "chunk_index": meta.get("chunk_index", 0),
                "total_chunks": meta.get("total_chunks", 0),
                "file_path": meta.get("file_path", ""),
            })

        await trace("subagents", "rag_results", f"Final RAG results: {len(output)} chunks", {
            "collection_id": collection_id,
            "query": query_text,
            "category_filter": category_filter,
            "output_count": len(output),
            "chunks": [
                {"chunk_id": o["chunk_id"], "doc_name": o["doc_name"], "score": round(o["score"], 4), "category": o["category"], "text_preview": o["text"][:200]}
                for o in output[:10]
            ],
        }, level="debug")

        return output


def _get_document_file_path(collection_id: str, doc_id: str) -> Path | None:
    """Helper to find document file path on disk."""
    doc_dir = DOCS_DIR / collection_id
    if not doc_dir.exists():
        return None
    for p in doc_dir.iterdir():
        if p.stem == doc_id:
            return p
    return None
