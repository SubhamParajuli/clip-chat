import asyncio
import sys
from langchain_core.messages import HumanMessage
from src.agent import async_app
from src.evaluate import run_evaluation  # Can be wrapped or updated for async
from src.ingest import ingest_video


def print_help():
  print("=" * 50)
  print(" Multi-Modal Agentic Video System - CLI Help ")
  print("=" * 50)
  print("Usage:")
  print("  python main.py ingest <video_path> <video_id>")
  print('  python main.py query "Your natural language question here"')
  print("  python main.py eval")
  print("=" * 50)


async def handle_query(query_text: str):
  """Asynchronously streams the LangGraph agent's reasoning and tool calls."""
  print(f"\nQuerying video system for: '{query_text}'\n" + "-" * 40)
  inputs = {"messages": [HumanMessage(content=query_text)]}

  try:
    # Use .astream() for non-blocking asynchronous execution
    async for chunk in async_app.astream(inputs, stream_mode="values"):
      latest_message = chunk["messages"][-1]
      latest_message.pretty_print()
  except Exception as e:
    print(f"Error during agent execution: {str(e)}")
  print("-" * 40)


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
    print(f"Starting ingestion process for {video_id} ({video_path})...")

    # Ingestion is heavy CPU bound (Whisper/OCR/CLIP). 
    # Run it in a thread executor to keep the async event loop unblocked if needed.
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, ingest_video, video_path, video_id)

  elif command == "query":
    if len(sys.argv) < 3:
      print("Error: Query text missing.")
      print('Example: python main.py query "When does the red car appear?"')
      sys.exit(1)

    query_text = " ".join(sys.argv[2:])
    await handle_query(query_text)

  elif command == "eval":
    print("Running evaluation pipeline...")
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, run_evaluation)

  else:
    print_help()


if __name__ == "__main__":
  # Run the async main event loop
  asyncio.run(main())