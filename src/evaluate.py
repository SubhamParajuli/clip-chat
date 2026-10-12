import json
import os
import re
from langchain_core.messages import HumanMessage
from src.agent import app


def extract_timestamp_from_text(text: str) -> float | None:
  """Helper regex to pull the first timestamp float found in the agent's response."""
  match = re.search(r"Timestamp:\s*([\d.]+)", text)
  if match:
    return float(match.group(1))
  return None


def run_evaluation(dataset_path: str = "eval_dataset.json", time_tolerance: float = 5.0):
  if not os.path.exists(dataset_path):
    print(f"Error: Evaluation dataset '{dataset_path}' not found.")
    return

  with open(dataset_path, "r") as f:
    test_cases = json.load(f)

  total_cases = len(test_cases)
  hit_count = 0
  temporal_passed = 0

  print(f"Starting Evaluation across {total_cases} test cases...\n" + "-" * 50)

  for i, test in enumerate(test_cases, 1):
    query = test["query"]
    exp_video = test["expected_video_id"]
    exp_start = test["expected_start"]

    print(f"Test [{i}/{total_cases}] Query: '{query}'")

    # Invoke the LangGraph agent app
    inputs = {"messages": [HumanMessage(content=query)]}
    try:
      final_state = app.invoke(inputs)
      agent_response = final_state["messages"][-1].content
    except Exception as e:
      print(f"  [FAIL] Agent execution error: {str(e)}")
      continue

    # 1. Check Retrieval Hit (Does response mention the correct video ID?)
    video_hit = exp_video in agent_response

    # 2. Check Temporal Accuracy (Is predicted timestamp within tolerance?)
    predicted_time = extract_timestamp_from_text(agent_response)
    time_hit = False

    if predicted_time is not None:
      time_diff = abs(predicted_time - exp_start)
      if time_diff <= time_tolerance:
        time_hit = True
        temporal_passed += 1
        print(f"  [PASS] Predicted time {predicted_time}s is within {time_tolerance}s of expected {exp_start}s.")
      else:
        print(f"  [FAIL] Time mismatch. Predicted: {predicted_time}s, Expected: {exp_start}s (Diff: {time_diff:.1f}s)")
    else:
      print(f"  [FAIL] Could not parse timestamp from agent response: '{agent_response}'")

    if video_hit and time_hit:
      hit_count += 1
    print("-" * 50)

  # Final Summary Metrics Report
  hit_rate = (hit_count / total_cases) * 100
  temporal_accuracy = (temporal_passed / total_cases) * 100

  print("\n" + "=" * 20 + " EVALUATION REPORT " + "=" * 20)
  print(f"Total Test Cases Evaluated : {total_cases}")
  print(f"Overall Success Rate       : {hit_rate:.1f}%")
  print(f"Temporal Accuracy (±{time_tolerance}s): {temporal_accuracy:.1f}%")
  print("=" * 59)