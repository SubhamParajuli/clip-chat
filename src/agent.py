import os
import subprocess
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from src.db_client import get_chroma_client, get_audio_collection, get_visual_collection
from src.logging_config import get_logger
import torch

logger = get_logger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info("Using compute device: %s", DEVICE)

# Global models
text_embedder = SentenceTransformer("all-MiniLM-L6-v2", device=DEVICE)
clip_model = SentenceTransformer("clip-ViT-B-32", device=DEVICE)


# 1. Enhanced Pydantic Schema including clip path output
class VideoSearchResponse(BaseModel):
    found_match: bool = Field(
        description=(
            "Set to True ONLY if the search results genuinely contain the"
            " requested content. Set to False if the content is missing or"
            " irrelevant."
        )
    )
    video_id: str = Field(
        default="", description="The video ID where the match was found."
    )
    start_time: float = Field(
        default=0.0, description="Start time of the clip in seconds."
    )
    end_time: float = Field(
        default=0.0, description="End time of the clip in seconds."
    )
    reasoning: str = Field(
        description=(
            "Explain why this timestamp matches, or explain clearly why no match"
            " was found in the video."
        )
    )
    clip_path: str = Field(
        default="",
        description="Path to the sliced clip file, or empty string if no match.",
    )


@tool
def search_video_database(query: str, target_video_id: str = None) -> str:
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
        query_embeddings=[visual_vector], n_results=2, where=where_filter
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


# 2. Helper FFmpeg Slicing Function with Absolute Path Resolution
def slice_video_segment(video_id: str, start_time: float, end_time: float) -> str:
    """Slices the source video file using FFmpeg and returns an absolute file path."""
    buffered_start = max(0.0, start_time - 2.0)
    buffered_end = end_time + 2.0

    # Fetch the exact source file path from ChromaDB metadata
    source_path = None
    try:
        client = get_chroma_client()
        audio_col = get_audio_collection(client)
        results = audio_col.get(where={"video_id": video_id}, limit=1)
        if results and results.get("metadatas") and len(results["metadatas"]) > 0:
            source_path = results["metadatas"][0].get("file_path")

        if not source_path:
            visual_col = get_visual_collection(client)
            v_results = visual_col.get(where={"video_id": video_id}, limit=1)
            if (
                v_results
                and v_results.get("metadatas")
                and len(v_results.get("metadatas")) > 0
            ):
                source_path = v_results["metadatas"][0].get("file_path")
    except Exception as e:
        logger.error("Failed to query file path from ChromaDB: %s", str(e))

    # Fallback check if path from DB doesn't exist or wasn't found
    if not source_path or not os.path.exists(source_path):
        for ext in [".mp4", ".mkv", ".mov"]:
            candidate = os.path.abspath(os.path.join("./data", f"{video_id}{ext}"))
            if os.path.exists(candidate):
                source_path = candidate
                break

    logger.info(
        "Attempting to slice video. Source path: %s | Start: %.2f | End: %.2f",
        source_path,
        buffered_start,
        buffered_end,
    )

    if not source_path or not os.path.exists(source_path):
        logger.warning("Source video path does not exist on disk: %s", source_path)
        return ""

    os.makedirs("./data", exist_ok=True)
    output_clip_name = f"{video_id}_clip_{int(buffered_start)}_{int(buffered_end)}.mp4"
    
    # Enforce absolute path to prevent static file mounting or rendering issues
    output_path = os.path.abspath(os.path.join("./data", output_clip_name))

    # Calculate duration instead of using absolute -to end time
    duration = buffered_end - buffered_start

    command = [
        "ffmpeg",
        "-y",
        "-ss",
        str(buffered_start),  # Input seeking (fast)
        "-i",
        source_path,
        "-t",
        str(duration),        # Duration of the clip
        "-c:v",
        "copy",               # Copy video stream without re-encoding (keeps it fast)
        "-c:a",
        "aac",                # Re-encode audio to AAC to ensure audio is never dropped
        "-strict",
        "experimental",
        output_path,
    ]

    try:
        subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
        )
        logger.info("Successfully generated clip with audio at: %s", output_path)
        return output_path
    except subprocess.CalledProcessError as e:
        error_output = e.stderr.decode("utf-8", errors="ignore")
        logger.error("FFmpeg failed. Error details:\n%s", error_output)
        return ""


AGENT_SYSTEM_PROMPT = """
You are a strict, precise video search validation agent. Your job is to verify whether search results genuinely match the user's query.

CRITICAL RULES:
1. NEVER force a match. If the search results do not clearly and directly contain the requested content, you MUST set `found_match` to `false`.
2. IGNORE audio chunks for visual queries (clothing, colors, physical appearance, visual actions) unless the visual frames explicitly confirm a match.
3. If you catch yourself writing phrases like "it is not a perfect match" or "it is the best match available", THAT IS A FAILURE. Set `found_match` to `false`.
4. When `found_match` is true, you MUST extract the exact Video ID (e.g., vid_119c6162f1bb) from the search tool output and put it into the `video_id` field.
5. If `found_match` is false, leave `video_id`, `start_time`, `end_time` empty/zero, and explain why in `reasoning`.
"""


# 3. LangGraph Nodes
async def call_model(state: MessagesState):
    system_prompt = SystemMessage(
        content=(
            AGENT_SYSTEM_PROMPT
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

    logger.info("Video agent query completed for video ID %s.", result.video_id)
    return result