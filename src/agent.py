import os 
import subprocess
from yt_dlp import YoutubeDL
from langchain_core.messages import SystemMessage
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

from langgraph.graph import StateGraph,START,END,MessagesState
from sentence_transformers import SentenceTransformer
from langgraph.prebuilt import ToolNode,tools_condition
from src.db_client import get_chroma_client,get_video_collection

text_embedder=SentenceTransformer("all-MiniLM-L6-v2")
clip_model=SentenceTransformer("clip-ViT-B-32")
collection=get_video_collection(get_chroma_client())

@tool
def download_youtube_video(url: str) -> str:
  """Downloads a YouTube video to the local data directory using its URL and returns the file path."""
  output_dir = "./data"
  os.makedirs(output_dir, exist_ok=True)

  ydl_opts = {
      "format": "best[ext=mp4]/best",
      "outtmpl": os.path.join(output_dir, "%(id)s.%(ext)s"),
      "quiet": True,
  }

  try:
    with YoutubeDL(ydl_opts) as ydl:
      info = ydl.extract_info(url, download=True)
      filename = ydl.prepare_filename(info)
      return (
          f"Successfully downloaded video. File path: {filename} | Video ID:"
          f" {info.get('id')}"
      )
  except Exception as e:
    return f"Failed to download video: {str(e)}"

@tool
def cut_video_clip(
    video_path: str, start_time: float, end_time: float, output_name: str = "clip.mp4"
) -> str:
  """Cuts a specific segment from a video file using start and end timestamps (in seconds) and saves it."""
  output_path = os.path.join("./data", output_name)

  # Construct FFmpeg command for fast, precise trimming
  # -ss: start time, -to: end time, -c copy: fast stream copy without re-encoding
  command = [
      "ffmpeg",
      "-y",
      "-ss",
      str(start_time),
      "-to",
      str(end_time),
      "-i",
      video_path,
      "-c",
      "copy",
      output_path,
  ]

  try:
    subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return (
        f"Successfully clipped video from {start_time}s to {end_time}s. Saved"
        f" to: {output_path}"
    )
  except subprocess.CalledProcessError as e:
    return f"Failed to cut video clip: {e.stderr.decode()}"

@tool
def search_video_database(query:str)->str:
    """Searches the multi-modal video database for visual description or dialogue."""
    query_vector=clip_model.encode(query).tolist()
    results=collection.query(query_embeddings=[query_vector],n_results=4)

    if not results["documents"][0]:
        return "No matching timestamps found in the video database."

    formatted_output=[]
    for doc,meta in zip(results["documents"][0],results["metadatas"][0]):
        formatted_output.append(
        f"- Video: {meta['video_id']} | Type: {meta['type'].upper()} | "
        f"Timestamp: {meta['start_time']}s - {meta['end_time']}s | Details: {doc}"
    )
    return "\n".join(formatted_output)


tools=[search_video_database]

llm=ChatOllama(model="llama3",temperature=0)
llm_with_tools=llm.bind_tools(tools)

##make it async

async def call_model(state:MessagesState):
    system_prompt=SystemMessage(
        content=("You are an advanced agentic video index assistant. Your job is to"
          " find exact timestamps when users ask about events, objects, or"
          " speech in videos. Use the search_video_database tool to discover"
          " matches and report precise time intervals.")
    )
    response=await llm_with_tools.ainvoke([system_prompt]+state["messages"])
    return{"messages":[response]}

workflow=StateGraph(MessagesState)
workflow.add_node("agent",call_model)
workflow.add_node("tools",ToolNode(tools))

workflow.add_edge(START,"agent")
workflow.add_conditional_edges("agent",tools_condition)
workflow.add_edge("tools","agent")

app=workflow.compile()

png_bytes=app.get_graph().draw_mermaid_png()
output_path="agent_grpah.png"
with open(output_path,"wb") as f:
    f.write(png_bytes)

print(f"Agent workflow graph successfully saved as {output_path}!")

    