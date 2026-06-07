"""Integration wrapper for DeepResearch-Bench upstream evaluation scripts.

Usage:
    uv run python -m benchmarks.deepresearch_bench.evaluate \
        --model-name deep-research-agent-mimo \
        --upstream ./benchmarks/deepresearch_bench/upstream

Prerequisites:
    - Reports already generated in results/<model_name>.jsonl
    - OPENROUTER_API_KEY or OPENAI_API_KEY env var set (for RACE)
    - JINA_API_KEY env var set (for FACT)
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


BENCH_DIR = Path(__file__).resolve().parent
UPSTREAM_DEFAULT = BENCH_DIR / "upstream"


def check_env():
    """Verify required API keys are present."""
    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    has_openrouter = bool(os.getenv("OPENROUTER_API_KEY"))
    has_ikun = bool(os.getenv("IKUN_API_KEY"))
    has_muci = bool(os.getenv("MICU_API_KEY"))
    has_mimo = bool(os.getenv("MIMO_API_KEY"))
    has_jina = bool(os.getenv("JINA_API_KEY"))

    if not (has_openai or has_openrouter or has_ikun or has_muci or has_mimo):
        print("WARNING: No LLM API key is set (OPENAI_API_KEY, OPENROUTER_API_KEY, IKUN_API_KEY, MICU_API_KEY, or MIMO_API_KEY).")
        print("RACE evaluation will fail.")
    if not has_jina:
        print("WARNING: JINA_API_KEY is not set.")
        print("FACT evaluation (web scraping) may fail or produce inaccurate results.")

    return (has_openai or has_openrouter or has_ikun or has_muci or has_mimo), has_jina


def run_race(model_name: str, upstream_dir: Path, output_dir: Path):
    """Run RACE evaluation via upstream script."""
    race_script = upstream_dir / "deepresearch_bench_race.py"
    raw_data_dir = upstream_dir / "data" / "test_data" / "raw_data"
    query_file = upstream_dir / "data" / "prompt_data" / "query.jsonl"
    race_output = output_dir / "race" / model_name
    race_output.mkdir(parents=True, exist_ok=True)

    # Copy our generated jsonl into upstream's expected location
    our_result = BENCH_DIR / "results" / f"{model_name}.jsonl"
    upstream_result = raw_data_dir / f"{model_name}.jsonl"

    if not our_result.exists():
        print(f"ERROR: Report file not found: {our_result}")
        print("Run the benchmark first:")
        print(f"  uv run python -m benchmarks.deepresearch_bench --model-name {model_name}")
        sys.exit(1)

    print(f"Copying report to upstream path: {upstream_result}")
    shutil.copy2(our_result, upstream_result)

    cmd = [
        sys.executable,
        str(race_script),
        model_name,
        "--raw_data_dir", str(raw_data_dir),
        "--max_workers", "10",
        "--query_file", str(query_file),
        "--output_dir", str(race_output),
    ]

    print(f"Running RACE evaluation...")
    print(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(upstream_dir))
    if result.returncode != 0:
        print(f"RACE evaluation failed with exit code {result.returncode}")
        return False

    print(f"RACE results saved to: {race_output / 'race_result.txt'}")
    return True


def run_fact(model_name: str, upstream_dir: Path, output_dir: Path):
    """Run FACT evaluation via upstream pipeline."""
    fact_output = output_dir / "fact" / model_name
    fact_output.mkdir(parents=True, exist_ok=True)

    raw_data_path = upstream_dir / "data" / "test_data" / "raw_data" / f"{model_name}.jsonl"
    query_data_path = upstream_dir / "data" / "prompt_data" / "query.jsonl"

    if not raw_data_path.exists():
        print(f"ERROR: Report file not found at upstream path: {raw_data_path}")
        sys.exit(1)

    steps = [
        ("Extracting citations", [
            sys.executable, "-m", "utils.extract",
            "--raw_data_path", str(raw_data_path),
            "--output_path", str(fact_output / "extracted.jsonl"),
            "--query_data_path", str(query_data_path),
            "--n_total_process", "10",
        ]),
        ("Deduplicating citations", [
            sys.executable, "-m", "utils.deduplicate",
            "--raw_data_path", str(fact_output / "extracted.jsonl"),
            "--output_path", str(fact_output / "deduplicated.jsonl"),
            "--query_data_path", str(query_data_path),
            "--n_total_process", "10",
        ]),
        ("Scraping webpages", [
            sys.executable, "-m", "utils.scrape",
            "--raw_data_path", str(fact_output / "deduplicated.jsonl"),
            "--output_path", str(fact_output / "scraped.jsonl"),
            "--n_total_process", "10",
        ]),
        ("Validating citations", [
            sys.executable, "-m", "utils.validate",
            "--raw_data_path", str(fact_output / "scraped.jsonl"),
            "--output_path", str(fact_output / "validated.jsonl"),
            "--query_data_path", str(query_data_path),
            "--n_total_process", "10",
        ]),
        ("Collecting statistics", [
            sys.executable, "-m", "utils.stat",
            "--input_path", str(fact_output / "validated.jsonl"),
            "--output_path", str(fact_output / "fact_result.txt"),
        ]),
    ]

    for desc, cmd in steps:
        print(f"\n==== {desc} ====")
        print(f"Command: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=str(upstream_dir))
        if result.returncode != 0:
            print(f"FACT step '{desc}' failed with exit code {result.returncode}")
            return False

    print(f"\nFACT results saved to: {fact_output / 'fact_result.txt'}")
    return True


def print_summary(model_name: str, output_dir: Path):
    """Print a concise summary of results."""
    race_file = output_dir / "race" / model_name / "race_result.txt"
    fact_file = output_dir / "fact" / model_name / "fact_result.txt"

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    if race_file.exists():
        print(f"\n--- RACE ({model_name}) ---")
        print(race_file.read_text(encoding="utf-8"))
    else:
        print(f"\nRACE results not found at {race_file}")

    if fact_file.exists():
        print(f"\n--- FACT ({model_name}) ---")
        print(fact_file.read_text(encoding="utf-8"))
    else:
        print(f"\nFACT results not found at {fact_file}")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Evaluate DeepResearch-Bench reports")
    parser.add_argument("--model-name", required=True, help="Model name used during generation")
    parser.add_argument("--upstream", type=str, default=str(UPSTREAM_DEFAULT), help="Path to upstream bench repo")
    parser.add_argument("--output-dir", type=str, default=str(BENCH_DIR / "results"), help="Output directory")
    parser.add_argument("--skip-race", action="store_true", help="Skip RACE evaluation")
    parser.add_argument("--skip-fact", action="store_true", help="Skip FACT evaluation")

    args = parser.parse_args()

    upstream_dir = Path(args.upstream)
    output_dir = Path(args.output_dir)

    if not upstream_dir.exists():
        print(f"ERROR: Upstream directory not found: {upstream_dir}")
        print("Make sure the submodule is initialized:")
        print("  git submodule update --init --recursive")
        sys.exit(1)

    has_llm, has_jina = check_env()

    if not args.skip_race and not has_llm:
        print("\nRACE requires OPENAI_API_KEY or OPENROUTER_API_KEY. Skipping RACE.")
        args.skip_race = True

    if not args.skip_fact and not has_jina:
        print("\nFACT requires JINA_API_KEY. Skipping FACT.")
        args.skip_fact = True

    success = True

    if not args.skip_race:
        success = run_race(args.model_name, upstream_dir, output_dir) and success

    if not args.skip_fact:
        success = run_fact(args.model_name, upstream_dir, output_dir) and success

    if success and not (args.skip_race and args.skip_fact):
        print_summary(args.model_name, output_dir)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
