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
    print(f"Error: Evaluation dataset '{dataset_path}' not found.")
    return

  with open(dataset_path, "r") as f:
    test_cases = json.load(f)

  total_cases = len(test_cases)
  match_correct_count = 0
  temporal_passed_count = 0
  negative_passed_count = 0
  negative_total = 0

  print(
      f"Starting Evaluation across {total_cases} test cases...\n" + "-" * 50
  )

  for i, test in enumerate(test_cases, 1):
    query = test["query"]
    exp_video = test["expected_video_id"]
    exp_start = test["expected_start"]
    should_exist = test["should_exist"]

    print(f"Test [{i}/{total_cases}] Query: '{query}'")

    try:
      # Invoke the production programmatic agent wrapper
      result = await query_video_agent_programmatic(query)
    except Exception as e:
      logger.exception("Evaluation agent execution failed for test %d.", i)
      print(f"  [FAIL] Agent execution error: {str(e)}")
      continue

    # Case A: Negative / Hallucination Test (Content should NOT exist)
    if not should_exist:
      negative_total += 1
      if not result.found_match:
        negative_passed_count += 1
        print("  [PASS] Successfully rejected non-existent content (Guardrail active).")
      else:
        print(
            f"  [FAIL] False positive! Agent claimed a match for non-existent query. Reasoning: {result.reasoning}"
        )
      print("-" * 50)
      continue

    # Case B: Positive Match Test (Content SHOULD exist)
    video_hit = result.found_match and (result.video_id == exp_video or not exp_video)
    
    if video_hit:
      match_correct_count += 1
      print(f"  [PASS] Correctly matched Video ID: {result.video_id}")
    else:
      print(
          f"  [FAIL] Video match failed. Expected ID: {exp_video}, Got ID: {result.video_id}, Found Match: {result.found_match}"
      )

    # Check Temporal Accuracy
    time_diff = abs(result.start_time - exp_start)
    if result.found_match and time_diff <= time_tolerance:
      temporal_passed_count += 1
      print(
          f"  [PASS] Start time {result.start_time}s is within {time_tolerance}s of expected {exp_start}s."
      )
    elif result.found_match:
      print(
          f"  [FAIL] Time mismatch. Predicted: {result.start_time}s, Expected: {exp_start}s (Diff: {time_diff:.1f}s)"
      )
    else:
      print("  [FAIL] Skipped temporal check because agent set found_match to False.")

    print("-" * 50)

  # Final Summary Metrics Report
  positive_total = total_cases - negative_total
  retrieval_accuracy = (match_correct_count / positive_total * 100) if positive_total > 0 else 0
  temporal_accuracy = (temporal_passed_count / positive_total * 100) if positive_total > 0 else 0
  guardrail_accuracy = (negative_passed_count / negative_total * 100) if negative_total > 0 else 100

  print("\n" + "=" * 20 + " EVALUATION REPORT " + "=" * 20)
  print(f"Total Test Cases Evaluated   : {total_cases}")
  print(f"Positive Match Accuracy      : {retrieval_accuracy:.1f}%")
  print(f"Temporal Accuracy (±{time_tolerance}s) : {temporal_accuracy:.1f}%")
  print(f"Guardrail / Hallucination Rejection : {guardrail_accuracy:.1f}%")
  print("=" * 59)


import asyncio
import sys

if __name__ == "__main__":
  if len(sys.argv) > 1 and sys.argv[1] == "eval":
    from evaluate import run_evaluation

    # Correctly run the async evaluation coroutine
    asyncio.run(run_evaluation())
  else:
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)