"""Benchmark script: vector vs BM25 vs hybrid retrieval recall comparison.

Usage:
    uv run python scripts/benchmark_retrieval.py

Measures recall@k for each retrieval strategy on a synthetic Chinese corpus.
"""

import asyncio
import tempfile
from pathlib import Path

import bm25s
import jieba


# ── synthetic corpus ────────────────────────────────────────────

CORPUS = [
    # Relevant to query 1: "深度学习在图像识别中的应用"
    "深度学习是机器学习的一个重要分支，主要基于多层神经网络结构。",
    "卷积神经网络（CNN）在图像识别任务中取得了突破性进展。",
    "计算机视觉领域广泛使用深度学习模型进行目标检测和分类。",
    "图像识别技术已被应用于医疗影像诊断和自动驾驶系统。",
    "ResNet 和 VGG 是两种经典的深度神经网络架构。",
    # Relevant to query 2: "自然语言处理中的Transformer模型"
    "Transformer 模型通过自注意力机制彻底改变了自然语言处理领域。",
    "BERT 和 GPT 都是基于 Transformer 架构的预训练语言模型。",
    "注意力机制允许模型在处理序列时关注输入的不同部分。",
    "机器翻译、文本摘要和问答系统都受益于 Transformer 技术。",
    "多头注意力是 Transformer 的核心组件之一。",
    # Distractor documents (not relevant to either query)
    "数据库索引技术对于提升查询性能至关重要。B+树是最常用的索引结构。",
    "云计算提供了弹性伸缩的计算资源，降低了企业的IT基础设施成本。",
    "区块链技术通过去中心化的分布式账本实现了数据不可篡改。",
    "操作系统的内存管理包括分页、分段和虚拟内存等关键技术。",
    "软件工程中的敏捷开发方法强调迭代交付和持续反馈。",
]

QUERIES = [
    ("深度学习在图像识别中的应用", {0, 1, 2, 3, 4}),
    ("自然语言处理中的Transformer模型", {5, 6, 7, 8, 9}),
]

K_VALUES = [3, 5, 10]


def _tokenize_chinese(text: str) -> list[str]:
    return list(jieba.cut(text))


def _rrf_fusion(vector_results, bm25_results, k=60):
    scores = {}
    for chunk_id, rank in vector_results:
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    for chunk_id, rank in bm25_results:
        scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.keys(), key=lambda x: scores[x], reverse=True)


async def benchmark():
    from sentence_transformers import SentenceTransformer
    import chromadb

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        chroma_dir = tmp_path / "chroma"
        client = chromadb.PersistentClient(path=str(chroma_dir))
        collection = client.create_collection(name="benchmark")

        model = SentenceTransformer("BAAI/bge-small-zh-v1.5")
        embeddings = model.encode(CORPUS)

        chunk_ids = [f"chunk_{i}" for i in range(len(CORPUS))]
        collection.add(
            ids=chunk_ids,
            documents=CORPUS,
            embeddings=embeddings.tolist(),
        )

        # Build BM25 index
        tokenized_corpus = bm25s.tokenize([
            " ".join(_tokenize_chinese(doc)) for doc in CORPUS
        ])
        retriever = bm25s.BM25()
        retriever.index(tokenized_corpus)

        results = {k: {"vector": 0.0, "bm25": 0.0, "hybrid": 0.0} for k in K_VALUES}

        for query_text, relevant_ids in QUERIES:
            # Vector search
            q_emb = model.encode([query_text])
            vec_res = collection.query(
                query_embeddings=q_emb.tolist(),
                n_results=10,
                include=["documents"],
            )
            vec_ranking = {int(cid.split("_")[1]): rank for rank, cid in enumerate(vec_res["ids"][0], start=1)}

            # BM25 search
            query_tokens = bm25s.tokenize(" ".join(_tokenize_chinese(query_text)))
            bm25_res, _ = retriever.retrieve(query_tokens, k=10)
            bm25_ranking = {bm25_res[0][rank]: rank + 1 for rank in range(len(bm25_res[0]))}

            # Hybrid (RRF)
            vec_list = [(f"chunk_{i}", r) for i, r in vec_ranking.items()]
            bm25_list = [(f"chunk_{i}", r) for i, r in bm25_ranking.items()]
            hybrid_order = _rrf_fusion(vec_list, bm25_list)
            hybrid_ranking = {int(cid.split("_")[1]): rank for rank, cid in enumerate(hybrid_order, start=1)}

            for k in K_VALUES:
                vec_top = set(i for i, r in vec_ranking.items() if r <= k)
                bm25_top = set(i for i, r in bm25_ranking.items() if r <= k)
                hybrid_top = set(i for i, r in hybrid_ranking.items() if r <= k)

                results[k]["vector"] += len(vec_top & relevant_ids) / len(relevant_ids)
                results[k]["bm25"] += len(bm25_top & relevant_ids) / len(relevant_ids)
                results[k]["hybrid"] += len(hybrid_top & relevant_ids) / len(relevant_ids)

        num_queries = len(QUERIES)
        print("=" * 60)
        print("Retrieval Benchmark: Vector vs BM25 vs Hybrid (RRF)")
        print("=" * 60)
        print(f"{'Metric':<12} {'Vector':>10} {'BM25':>10} {'Hybrid':>10}")
        print("-" * 60)
        for k in K_VALUES:
            print(
                f"Recall@{k:<5}  "
                f"{results[k]['vector'] / num_queries:>10.3f} "
                f"{results[k]['bm25'] / num_queries:>10.3f} "
                f"{results[k]['hybrid'] / num_queries:>10.3f}"
            )
        print("=" * 60)


if __name__ == "__main__":
    asyncio.run(benchmark())
