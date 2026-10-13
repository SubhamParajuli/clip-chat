import asyncio
import json
import os
from src.agent import query_video_agent_programmatic
from src.logging_config import get_logger

logger = get_logger(__name__)


async def run_evaluation(
    dataset_path: str = "eval_dataset.json", time_tolerance: float = 5.0
):
    if not os.path.exists(dataset_path):
        logger.error("Evaluation dataset '%s' not found.", dataset_path)
        return {
            "error": f"Evaluation dataset '{dataset_path}' not found.",
            "total_cases": 0,
        }

    with open(dataset_path, "r") as f:
        test_cases = json.load(f)

    total_cases = len(test_cases)
    match_correct_count = 0
    temporal_passed_count = 0
    negative_passed_count = 0
    negative_total = 0

    logger.info("Starting Evaluation across %d test cases...", total_cases)

    for i, test in enumerate(test_cases, 1):
        query = test["query"]
        exp_video = test["expected_video_id"]
        exp_start = test["expected_start"]
        should_exist = test["should_exist"]

        logger.info("Test [%d/%d] Query: '%s'", i, total_cases, query)

        try:
            # Invoke the production programmatic agent wrapper
            result = await query_video_agent_programmatic(query)
        except Exception as e:
            logger.exception("Evaluation agent execution failed for test %d.", i)
            logger.error("  [FAIL] Agent execution error: %s", str(e))
            continue

        # Case A: Negative / Hallucination Test (Content should NOT exist)
        if not should_exist:
            negative_total += 1
            if not result.found_match:
                negative_passed_count += 1
                logger.info("  [PASS] Successfully rejected non-existent content (Guardrail active).")
            else:
                logger.warning(
                    "  [FAIL] False positive! Agent claimed a match for non-existent query. Reasoning: %s",
                    result.reasoning,
                )
            continue

        # Case B: Positive Match Test (Content SHOULD exist)
        video_hit = result.found_match and (result.video_id == exp_video or not exp_video)
        
        if video_hit:
            match_correct_count += 1
            logger.info("  [PASS] Correctly matched Video ID: %s", result.video_id)
        else:
            logger.warning(
                "  [FAIL] Video match failed. Expected ID: %s, Got ID: %s, Found Match: %s",
                exp_video,
                result.video_id,
                result.found_match,
            )

        # Check Temporal Accuracy
        time_diff = abs(result.start_time - exp_start)
        if result.found_match and time_diff <= time_tolerance:
            temporal_passed_count += 1
            logger.info(
                "  [PASS] Start time %ss is within %ss of expected %ss.",
                result.start_time,
                time_tolerance,
                exp_start,
            )
        elif result.found_match:
            logger.warning(
                "  [FAIL] Time mismatch. Predicted: %ss, Expected: %ss (Diff: %.1fs)",
                result.start_time,
                exp_start,
                time_diff,
            )
        else:
            logger.warning("  [FAIL] Skipped temporal check because agent set found_match to False.")

    # Final Summary Metrics Report
    positive_total = total_cases - negative_total
    retrieval_accuracy = (match_correct_count / positive_total * 100) if positive_total > 0 else 0
    temporal_accuracy = (temporal_passed_count / positive_total * 100) if positive_total > 0 else 0
    guardrail_accuracy = (negative_passed_count / negative_total * 100) if negative_total > 0 else 100

    metrics = {
        "total_cases": total_cases,
        "positive_total": positive_total,
        "negative_total": negative_total,
        "match_correct_count": match_correct_count,
        "temporal_passed_count": temporal_passed_count,
        "negative_passed_count": negative_passed_count,
        "retrieval_accuracy_percent": round(retrieval_accuracy, 1),
        "temporal_accuracy_percent": round(temporal_accuracy, 1),
        "guardrail_accuracy_percent": round(guardrail_accuracy, 1),
        "time_tolerance_seconds": time_tolerance
    }

    logger.info("EVALUATION REPORT:\n%s", json.dumps(metrics, indent=2))
    return metrics


import sys

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "eval":
        from evaluate import run_evaluation

        # Correctly run the async evaluation coroutine and handle output
        asyncio.run(run_evaluation())
    else:
        import uvicorn

        uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)