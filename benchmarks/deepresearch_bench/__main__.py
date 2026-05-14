"""CLI entry point for DeepResearch-Bench runner."""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from runner import run_benchmark


def main():
    parser = argparse.ArgumentParser(description="Run DeepResearch-Bench tasks")
    parser.add_argument("--model-name", default="deep-research-agent", help="Model name for output file")
    parser.add_argument("--n-zh", type=int, default=3, help="Number of Chinese tasks to sample")
    parser.add_argument("--n-en", type=int, default=3, help="Number of English tasks to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--max-iterations", type=int, default=2, help="Max reflection iterations")
    parser.add_argument("--quality-threshold", type=float, default=0.65, help="Quality threshold")
    parser.add_argument("--timeout", type=int, default=3600, help="Per-task timeout in seconds")

    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else None

    asyncio.run(run_benchmark(
        model_name=args.model_name,
        n_zh=args.n_zh,
        n_en=args.n_en,
        seed=args.seed,
        output_dir=output_dir,
        max_iterations=args.max_iterations,
        quality_threshold=args.quality_threshold,
        timeout=args.timeout,
    ))


if __name__ == "__main__":
    main()
