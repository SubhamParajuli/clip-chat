import asyncio
import os
import streamlit as st
from src.agent import query_video_agent_programmatic
from src.ingest import ingest_from_source

st.set_page_config(
    page_title="Multi-Modal Video Search Agent", layout="centered"
)

st.title("Local AI Multi-Modal Video Search Agent")
st.markdown(
    "Search your video library using natural language, powered by LangGraph, "
    "ChromaDB, Whisper, and CLIP."
)

# Sidebar Ingestion Section
st.sidebar.header("Ingest Video")
ingest_type = st.sidebar.radio("Source Type", ["YouTube URL", "Upload File"])

custom_id = st.sidebar.text_input(
    "Custom Video ID (Optional)", placeholder="Leave blank for auto-hashing"
)

if ingest_type == "YouTube URL":
  source_url = st.sidebar.text_input(
      "YouTube URL", placeholder="https://www.youtube.com/watch?v=..."
  )
  if st.sidebar.button("Ingest YouTube Video"):
    if source_url:
      with st.spinner("Downloading and processing YouTube video..."):
        try:
          vid_id = ingest_from_source(
              source_url, custom_id if custom_id else None
          )
          st.sidebar.success(f"Successfully ingested ID: {vid_id}")
        except Exception as e:
          st.sidebar.error(f"Error: {e}")
    else:
      st.sidebar.warning("Please provide a YouTube URL.")

else:
  uploaded_file = st.sidebar.file_uploader(
      "Choose a video file", type=["mp4", "mov", "mkv"]
  )
  if st.sidebar.button("Upload & Ingest File"):
    if uploaded_file is not None:
      os.makedirs("./data", exist_ok=True)
      file_path = os.path.join("./data", uploaded_file.name)

      with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

      with st.spinner(
          "Processing video content (Whisper, OCR, CLIP vectors)..."
      ):
        try:
          vid_id = ingest_from_source(
              file_path, custom_id if custom_id else None
          )
          st.sidebar.success(f"Successfully ingested ID: {vid_id}")
        except Exception as e:
          st.sidebar.error(f"Error: {e}")
    else:
      st.sidebar.warning("Please select a file to upload.")

# Main Search Section
st.header("Search Video Library")
query_input = st.text_input(
    "What are you looking for?",
    placeholder="e.g., When does the speaker show code?",
)

if st.button("Search & Cut Clip"):
  if query_input:
    with st.spinner("Agent is reasoning and checking relevance guardrails..."):
      result = asyncio.run(query_video_agent_programmatic(query_input))

      # Guardrail check
      if not result.found_match:
        st.warning(
            "Guardrail Notice: The agent searched the video database "
            "but found no authentic match for your query."
        )
        st.info(f"Agent Reasoning: {result.reasoning}")
      else:
        st.success("Match Found and Verified")
        col1, col2 = st.columns(2)

        with col1:
          st.metric("Video ID", result.video_id)
          st.metric("Start Time", f"{result.start_time}s")
          st.metric("End Time", f"{result.end_time}s")

        with col2:
          st.markdown(f"**Reasoning:** {result.reasoning}")

        if result.clip_path and os.path.exists(result.clip_path):
          st.subheader("Generated Video Clip Preview")
          st.video(result.clip_path)
        else:
          st.warning("Match verified, but clip file path is missing.")
  else:
    st.warning("Please enter a search query.")