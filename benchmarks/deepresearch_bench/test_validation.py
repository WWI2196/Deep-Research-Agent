"""Quick benchmark validation test - run 1-2 tasks to verify improvements."""

import asyncio
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from backend.config import get_config
from backend.graph import build_and_run_graph
from backend.persistence import init_db

init_db()


test_tasks = [
    {
        "id": 2,
        "language": "zh",
        "prompt": "收集整理目前国际综合实力前十的保险公司的相关资料，横向比较各公司的融资情况、信誉度、过往五年的增长幅度、实际分红、未来在中国发展潜力等维度，并为我评估出最有可能在未来资产排名靠前的2-3家公司",
    },
]


async def _noop_event(evt: dict) -> None:
    pass


async def run_task(task: dict) -> dict:
    print(f"\n{'='*60}")
    print(f"Task {task['id']}: {task['prompt'][:60]}...")
    print(f"{'='*60}")

    cfg = get_config()
    state = {
        "user_query": task["prompt"],
        "run_id": f"test_{task['id']}",
        "events": [],
        "errors": [],
        "max_iterations": 2,
        "quality_threshold": 0.65,
        "context_compress_retries": 1,
        "keep_tool_results": 3,
        "document_collections": [],
        "output_language": task["language"],
        "bench_format": True,
    }

    try:
        final_state = await asyncio.wait_for(
            build_and_run_graph(state, _noop_event),
            timeout=1800,
        )
        report = final_state.get("cited_report") or final_state.get("report", "")

        # Check requirements extraction
        requirements = final_state.get("requirements", {})
        print(f"\n📋 Extracted Requirements:")
        print(json.dumps(requirements, ensure_ascii=False, indent=2))

        # Check task compliance in report
        print(f"\n🔍 Task Compliance Check:")
        prompt_text = task["prompt"]
        report_lower = report.lower()

        # Check for comparison
        comparison_keywords = ["比较", "对比", "comparison", "vs", "横向"]
        has_comparison = any(kw in report_lower for kw in comparison_keywords)
        print(f"  Comparison (比较/对比): {'✅' if has_comparison else '❌'}")

        # Check for recommendation/evaluation
        recommend_keywords = ["推荐", "建议", "评估", "recommend", "排名", "前", "最优"]
        has_recommendation = any(kw in report_lower for kw in recommend_keywords)
        print(f"  Recommendation (推荐/评估): {'✅' if has_recommendation else '❌'}")

        # Check for specific count (2-3 companies)
        import re
        has_specific_count = bool(re.search(r'2[-–—]3|两三家|两到三家', report))
        print(f"  Specific count (2-3家): {'✅' if has_specific_count else '❌'}")

        # Check for dimensions mentioned
        dimensions = ["融资", "信誉", "增长", "分红", "潜力"]
        found_dims = [d for d in dimensions if d in report]
        print(f"  Dimensions covered ({len(found_dims)}/5): {', '.join(found_dims)}")

        # Check scope constraints
        has_china = "中国" in report or "china" in report_lower
        print(f"  Scope - China focus: {'✅' if has_china else '❌'}")

        print(f"\n📄 Report length: {len(report)} chars")
        print(f"📄 Report preview (first 500 chars):\n{report[:500]}...")

        return {
            "id": task["id"],
            "status": "success",
            "report_length": len(report),
            "requirements": requirements,
            "compliance": {
                "comparison": has_comparison,
                "recommendation": has_recommendation,
                "specific_count": has_specific_count,
                "dimensions": len(found_dims),
                "china_scope": has_china,
            },
        }

    except asyncio.TimeoutError:
        print("❌ Timeout (>30 min)")
        return {"id": task["id"], "status": "timeout"}
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return {"id": task["id"], "status": "error", "error": str(e)}


async def main():
    print("Starting small sample benchmark test...")
    print(f"Testing {len(test_tasks)} task(s)")

    results = []
    for task in test_tasks:
        result = await run_task(task)
        results.append(result)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for r in results:
        status_icon = "✅" if r["status"] == "success" else "❌"
        print(f"{status_icon} Task {r['id']}: {r['status']}")
        if r["status"] == "success":
            print(f"   Report: {r['report_length']} chars")
            print(f"   Compliance: {json.dumps(r['compliance'], ensure_ascii=False)}")


if __name__ == "__main__":
    asyncio.run(main())
