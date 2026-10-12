import asyncio
import sys
from src.agent import query_video_agent_programmatic
from src.evaluate import run_evaluation
from src.ingest import ingest_video


def print_help():
  print("=" * 60)
  print(" Multi-Modal Agentic Video System - CLI Runner ")
  print("=" * 60)
  print("Commands:")
  print("  1. Ingest Video:")
  print("     python main.py ingest <video_path> <video_id>")
  print("     Example: python main.py ingest data/sample.mp4 vid_01")
  print("\n  2. Query Agent & Generate Clip:")
  print('     python main.py query "Your natural language question"')
  print(
      '     Example: python main.py query "When does the red car with number'
      ' 1234 appear?"'
  )
  print("\n  3. Run Evaluation:")
  print("     python main.py eval")
  print("=" * 60)


async def main():
  if len(sys.argv) < 2:
    print_help()
    sys.exit(1)

  command = sys.argv[1].lower()

  if command == "ingest":
    if len(sys.argv) != 4:
      print("Error: Ingest requires a video path and a video ID.")
      print("Example: python main.py ingest data/sample.mp4 vid_01")
      sys.exit(1)

    video_path = sys.argv[2]
    video_id = sys.argv[3]
    print(f"Starting optimized ingestion for video: {video_id} ({video_path})")

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, ingest_video, video_path, video_id)
    print("Ingestion complete and saved to local ChromaDB!")

  elif command == "query":
    if len(sys.argv) < 3:
      print("Error: Query text missing.")
      print('Example: python main.py query "When does the red car appear?"')
      sys.exit(1)

    query_text = " ".join(sys.argv[2:])
    print(f"\nQuerying agent for: '{query_text}'...\n" + "-" * 50)

    # Run the agent programmatically; returns a validated Pydantic model with clip_path
    result = await query_video_agent_programmatic(query_text)

    print("=" * 20 + " STRUCTURED AGENT RESULT " + "=" * 20)
    print(f"Video ID   : {result.video_id}")
    print(f"Start Time : {result.start_time}s")
    print(f"End Time   : {result.end_time}s")
    print(f"Reasoning  : {result.reasoning}")
    if result.clip_path:
      print(f"Saved Clip : {result.clip_path} (Ready for playback!)")
    else:
      print(
          "Saved Clip : [Warning] Could not generate clip (Check if source"
          " video exists in ./data/)"
      )
    print("=" * 65)

  elif command == "eval":
    print("Running system evaluation pipeline...")
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, run_evaluation)

  else:
    print_help()


if __name__ == "__main__":
  asyncio.run(main())