"""DeepResearch-Bench runner — generates research reports for benchmark tasks."""

import asyncio
import json
import os
import random
import sys
import time
from pathlib import Path

# Ensure src/ is on PYTHONPATH when running from benchmarks/
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from backend.config import get_config
from backend.graph import build_and_run_graph
from backend.persistence import init_db
from backend.tracing import current_run_id

# Ensure DB tables exist when running outside the FastAPI server
init_db()


BENCH_DIR = Path(__file__).resolve().parent
UPSTREAM_DIR = BENCH_DIR / "upstream"
QUERY_PATH = UPSTREAM_DIR / "data" / "prompt_data" / "query.jsonl"

DEFAULT_OUTPUT_DIR = BENCH_DIR / "results"
DEFAULT_TIMEOUT = 3600  # 1 hour per task (bench itself has no limit)


def load_queries() -> list[dict]:
    """Load all 100 benchmark queries."""
    queries = []
    with open(QUERY_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(json.loads(line))
    return queries


def sample_queries(queries: list[dict], n_zh: int = 3, n_en: int = 3, seed: int = 42) -> list[dict]:
    """Randomly sample n_zh Chinese and n_en English tasks, preserving id order."""
    zh = [q for q in queries if q.get("language") == "zh"]
    en = [q for q in queries if q.get("language") == "en"]
    rng = random.Random(seed)
    sampled = rng.sample(zh, min(n_zh, len(zh))) + rng.sample(en, min(n_en, len(en)))
    sampled.sort(key=lambda q: q["id"])
    return sampled


async def _noop_event(evt: dict) -> None:
    pass


async def run_single_task(
    query: dict,
    max_iterations: int = 2,
    quality_threshold: float = 0.65,
    timeout: float = DEFAULT_TIMEOUT,
) -> dict:
    """Run one benchmark task through the research pipeline."""
    run_id = os.urandom(8).hex()
    cfg = get_config()
    language = query.get("language", "zh")

    state = {
        "user_query": query["prompt"],
        "run_id": run_id,
        "events": [],
        "errors": [],
        "max_iterations": max_iterations,
        "quality_threshold": quality_threshold,
        "context_compress_retries": cfg.context_compress_retries,
        "keep_tool_results": cfg.keep_tool_results,
        "document_collections": [],
        "output_language": language,
        "bench_format": True,
    }

    start = time.time()
    try:
        final_state = await asyncio.wait_for(
            build_and_run_graph(state, _noop_event),
            timeout=timeout,
        )
        article = final_state.get("cited_report") or final_state.get("report", "")
        status = "success"
    except asyncio.TimeoutError:
        article = "[TIMEOUT]"
        status = "timeout"
    except Exception as e:
        article = f"[ERROR: {e}]"
        status = "error"

    elapsed = time.time() - start
    return {
        "id": query["id"],
        "prompt": query["prompt"],
        "article": article,
        "status": status,
        "elapsed": round(elapsed, 2),
        "language": language,
    }


async def run_benchmark(
    model_name: str,
    n_zh: int = 3,
    n_en: int = 3,
    seed: int = 42,
    output_dir: Path | None = None,
    max_iterations: int = 2,
    quality_threshold: float = 0.65,
    timeout: float = DEFAULT_TIMEOUT,
) -> Path:
    """Run sampled tasks and write results in bench format."""
    output_dir = output_dir or DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{model_name}.jsonl"

    queries = load_queries()
    sampled = sample_queries(queries, n_zh=n_zh, n_en=n_en, seed=seed)

    print(f"DeepResearch-Bench runner")
    print(f"Model name : {model_name}")
    print(f"Tasks      : {len(sampled)} (zh={n_zh}, en={n_en})")
    print(f"Output     : {output_file}")
    print("-" * 50)

    # Load already-completed ids for resume support
    completed_ids = set()
    if output_file.exists():
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    completed_ids.add(json.loads(line)["id"])
        print(f"Resuming   : {len(completed_ids)} already done")

    results = []
    for q in sampled:
        if q["id"] in completed_ids:
            print(f"[SKIP] id={q['id']} (already done)")
            continue

        lang_label = "zh" if q.get("language") == "zh" else "en"
        print(f"[RUN ] id={q['id']} lang={lang_label} | {q['prompt'][:60]}...")

        result = await run_single_task(
            q,
            max_iterations=max_iterations,
            quality_threshold=quality_threshold,
            timeout=timeout,
        )
        results.append(result)

        # Append immediately for crash-resume
        with open(output_file, "a", encoding="utf-8") as f:
            out = {"id": result["id"], "prompt": result["prompt"], "article": result["article"]}
            f.write(json.dumps(out, ensure_ascii=False) + "\n")

        print(f"[DONE] id={result['id']} status={result['status']} time={result['elapsed']}s")
        print("-" * 50)

    print(f"\nFinished. Results written to {output_file}")
    return output_file
