import os
import subprocess
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from src.db_client import get_chroma_client, get_audio_collection,get_visual_collection
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using compute device: {DEVICE}")
# Global models
text_embedder = SentenceTransformer("all-MiniLM-L6-v2",device=DEVICE)
clip_model = SentenceTransformer("clip-ViT-B-32",device=DEVICE)



# 1. Enhanced Pydantic Schema including clip path output
class VideoSearchResponse(BaseModel):
  video_id: str = Field(
      description="The ID of the video where the event occurred."
  )
  start_time: float = Field(
      description="The exact start timestamp in seconds of the matching event."
  )
  end_time: float = Field(
      description="The exact end timestamp in seconds of the matching event."
  )
  reasoning: str = Field(
      description="Explanation of why this timestamp matches the user query."
  )
  clip_path: str = Field(
      default="", description="The local file path of the generated video clip."
  )

@tool
def search_video_database(query: str,target_video_id: str = None) -> str:
  """Searches both audio transcripts and visual frame collections for matches."""
  client = get_chroma_client()
  audio_col = get_audio_collection(client)
  visual_col = get_visual_collection(client)
  where_filter = {"video_id": target_video_id} if target_video_id else None

  # Query audio collection (uses 384-dim text embedder)
  audio_vector = text_embedder.encode(query).tolist()
  audio_results = audio_col.query(query_embeddings=[audio_vector], n_results=2)

  # Query visual collection (uses 512-dim CLIP model for text-to-image semantic matching)
  visual_vector = clip_model.encode(query).tolist()
  visual_results = visual_col.query(
      query_embeddings=[visual_vector], n_results=2,where=where_filter
  )

  formatted_output = []

  if audio_results["documents"] and audio_results["documents"][0]:
    for doc, meta in zip(
        audio_results["documents"][0], audio_results["metadatas"][0]
    ):
      formatted_output.append(
          f"- Video: {meta['video_id']} | Type: AUDIO | "
          f"Timestamp: {meta['start_time']}s - {meta['end_time']}s | Speech:"
          f" {doc}"
      )

  if visual_results["documents"] and visual_results["documents"][0]:
    for doc, meta in zip(
        visual_results["documents"][0], visual_results["metadatas"][0]
    ):
      formatted_output.append(
          f"- Video: {meta['video_id']} | Type: VISUAL | "
          f"Timestamp: {meta['start_time']}s - {meta['end_time']}s | Details:"
          f" {doc}"
      )

  if not formatted_output:
    return "No matching timestamps found in the video database."

  return "\n".join(formatted_output)

tools = [search_video_database]
llm = ChatOllama(model="llama3.1:8b", temperature=0)
llm_with_tools = llm.bind_tools(tools)
structured_llm = llm.with_structured_output(VideoSearchResponse)


# 2. Helper FFmpeg Slicing Function
def slice_video_segment(video_id: str, start_time: float, end_time: float) -> str:
  """Unconditionally slices the source video file using FFmpeg

  with a natural buffer window for better context.
  """
  # --- 2nd IMPROVEMENT: SMART CLIP PADDING ---
  # Add a 2-second buffer before and after the match
  # max(0.0, ...) ensures the start time never goes below 0 seconds
  buffered_start = max(0.0, start_time - 2.0)
  buffered_end = end_time + 2.0
  # -------------------------------------------

  # Locate source video file in the data folder
  source_path = None
  for ext in [".mp4", ".mkv", ".mov"]:
    candidate = os.path.join("./data", f"{video_id}{ext}")
    if os.path.exists(candidate):
      source_path = candidate
      break

  if not source_path:
    return ""  # Source video file not found locally

  output_clip_name = f"{video_id}_clip_{int(buffered_start)}_{int(buffered_end)}.mp4"
  output_path = os.path.join("./data", output_clip_name)

  # FFmpeg command using the buffered start and end times
  command = [
      "ffmpeg",
      "-y",
      "-ss",
      str(buffered_start),
      "-to",
      str(buffered_end),
      "-i",
      source_path,
      "-c",
      "copy",
      output_path,
  ]

  try:
    subprocess.run(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
    )
    return output_path
  except subprocess.CalledProcessError:
    return ""

# 3. LangGraph Nodes
async def call_model(state: MessagesState):
  system_prompt = SystemMessage(
      content=(
          "You are an advanced agentic video search assistant. Use the"
          " search_video_database tool to find exact matching timestamps for"
          " the user request."
      )
  )
  response = await llm_with_tools.ainvoke([system_prompt] + state["messages"])
  return {"messages": [response]}


async def format_structured_output(state: MessagesState):
  messages = state["messages"]
  context_str = "\n".join(
      [
          msg.content
          for msg in messages
          if isinstance(msg, (ToolMessage, HumanMessage))
      ]
  )

  prompt = (
      "Based on the following database search results, extract the exact video"
      " ID, start time, end time, and reasoning matching the user's initial"
      f" intent.\n\nContext:\n{context_str}"
  )

  structured_result: VideoSearchResponse = await structured_llm.ainvoke(prompt)
  return {"messages": [HumanMessage(content=structured_result.model_dump_json())]}


# 4. Build LangGraph Workflow
workflow = StateGraph(MessagesState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", ToolNode(tools))
workflow.add_node("formatter", format_structured_output)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges(
    "agent",
    lambda state: "tools" if state["messages"][-1].tool_calls else "formatter",
)
workflow.add_edge("tools", "agent")
workflow.add_edge("formatter", END)

async_app = workflow.compile()


# 5. Unconditional Programmatic Execution Wrapper
async def query_video_agent_programmatic(query_text: str) -> VideoSearchResponse:
  """Runs the graph, extracts structured metadata, and unconditionally cuts

  and returns the physical video clip path.
  """
  inputs = {"messages": [HumanMessage(content=query_text)]}
  final_state = await async_app.ainvoke(inputs)

  # Parse initial structured response from the graph
  final_content = final_state["messages"][-1].content
  result = VideoSearchResponse.model_validate_json(final_content)

  # UNCONDITIONAL CLIP EXTRACTION: Automatically slice the clip using FFmpeg
  if result.video_id and result.start_time is not None:
    clip_file_path = slice_video_segment(
        result.video_id, result.start_time, result.end_time
    )
    result.clip_path = clip_file_path

  return result