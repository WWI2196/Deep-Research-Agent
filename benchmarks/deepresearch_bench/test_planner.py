"""Quick planner validation - test requirements extraction only."""

import asyncio
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from backend.planning import generate_research_plan


test_queries = [
    {
        "id": 2,
        "query": "收集整理目前国际综合实力前十的保险公司的相关资料，横向比较各公司的融资情况、信誉度、过往五年的增长幅度、实际分红、未来在中国发展潜力等维度，并为我评估出最有可能在未来资产排名靠前的2-3家公司",
        "expected_objectives": ["横向比较", "评估", "推荐"],
    },
    {
        "id": 3,
        "query": "中国金融未来的发展趋势，未来哪一个细分领域（例如投行、pe、固收等）更有上升空间",
        "expected_objectives": ["预测", "分析", "推荐"],
    },
]


async def test_planner(query_info):
    print(f"\n{'='*60}")
    print(f"Task {query_info['id']}: {query_info['query'][:50]}...")
    print(f"{'='*60}")
    
    try:
        plan = await asyncio.wait_for(
            generate_research_plan(query_info["query"]),
            timeout=120,
        )
        
        print(f"\n📋 Extracted Requirements:")
        requirements = plan.get("requirements", {})
        print(json.dumps(requirements, ensure_ascii=False, indent=2))
        
        print(f"\n🔍 Validation:")
        core_objs = requirements.get("core_objectives", [])
        print(f"  Core objectives: {core_objs}")
        
        # Check expected objectives
        found_expected = 0
        for expected in query_info["expected_objectives"]:
            found = any(expected in obj for obj in core_objs)
            if found:
                found_expected += 1
                print(f"  ✅ Found expected objective: {expected}")
            else:
                print(f"  ❌ Missing expected objective: {expected}")
        
        # Check explicit requirements
        explicit = requirements.get("explicit_requirements", [])
        print(f"\n  Explicit requirements ({len(explicit)}):")
        for req in explicit:
            print(f"    - {req}")
        
        # Check scope constraints
        scope = requirements.get("scope_constraints", {})
        print(f"\n  Scope constraints:")
        for k, v in scope.items():
            if v:
                print(f"    ✅ {k}: {v}")
            else:
                print(f"    ⚠️  {k}: (not detected)")
        
        # Check dimensions
        dims = plan.get("dimensions", [])
        print(f"\n  Dimensions: {len(dims)}")
        for d in dims:
            print(f"    - {d.get('name', 'N/A')}: {d.get('scope', '')[:60]}...")
        
        return {
            "id": query_info["id"],
            "status": "success",
            "objectives_found": found_expected,
            "objectives_total": len(query_info["expected_objectives"]),
            "dimensions": len(dims),
            "requirements": requirements,
        }
        
    except asyncio.TimeoutError:
        print("❌ Timeout")
        return {"id": query_info["id"], "status": "timeout"}
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return {"id": query_info["id"], "status": "error", "error": str(e)}


async def main():
    print("Testing planner requirements extraction...")
    
    results = []
    for query_info in test_queries:
        result = await test_planner(query_info)
        results.append(result)
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for r in results:
        if r["status"] == "success":
            print(f"✅ Task {r['id']}: {r['objectives_found']}/{r['objectives_total']} objectives detected, {r['dimensions']} dimensions")
        else:
            print(f"❌ Task {r['id']}: {r['status']}")


if __name__ == "__main__":
    asyncio.run(main())
