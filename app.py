import os
import requests
import streamlit as st

API_BASE_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Multi-Modal Video Search Agent", layout="centered"
)

st.title("Clip-Search : Multi-Modal Video Search Agent")
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
      with st.spinner("Downloading and processing YouTube video via API..."):
        try:
          payload = {"source": source_url}
          if custom_id:
            payload["video_id"] = custom_id

          response = requests.post(f"{API_BASE_URL}/ingest-url", json=payload)
          if response.status_code == 200:
            data = response.json()
            st.sidebar.success(
                f"Successfully ingested ID: {data.get('video_id')}"
            )
          else:
            st.sidebar.error(f"Error: {response.json().get('detail')}")
        except Exception as e:
          st.sidebar.error(f"Connection Error: {e}")
    else:
      st.sidebar.warning("Please provide a YouTube URL.")

else:
  uploaded_file = st.sidebar.file_uploader(
      "Choose a video file", type=["mp4", "mov", "mkv"]
  )
  if st.sidebar.button("Upload & Ingest File"):
    if uploaded_file is not None:
      with st.spinner("Uploading and processing video content..."):
        try:
          files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
          data = {"video_id": custom_id} if custom_id else {}

          response = requests.post(
              f"{API_BASE_URL}/upload-video", files=files, data=data
          )
          if response.status_code == 200:
            res_data = response.json()
            st.sidebar.success(
                f"Successfully ingested ID: {res_data.get('video_id')}"
            )
          else:
            st.sidebar.error(f"Error: {response.json().get('detail')}")
        except Exception as e:
          st.sidebar.error(f"Connection Error: {e}")
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
      try:
        response = requests.post(
            f"{API_BASE_URL}/search", params={"query": query_input}
        )
        if response.status_code != 200:
          st.error(f"Search API Error: {response.json().get('detail')}")
        else:
          result = response.json()

          # Guardrail check
          if not result.get("found_match", False):
            st.warning(
                "Guardrail Notice: The agent searched the video database "
                "but found no authentic match for your query."
            )
            st.info(f"Agent Reasoning: {result.get('reasoning')}")
          else:
            st.success("Match Found and Verified")
            col1, col2 = st.columns(2)

            with col1:
              st.metric("Video ID", result.get("video_id"))
              st.metric("Start Time", f"{result.get('start_time')}s")
              st.metric("End Time", f"{result.get('end_time')}s")

            with col2:
              st.markdown(f"**Reasoning:** {result.get('reasoning')}")

            clip_path = result.get("clip_path", "")
            if clip_path:
              # Convert absolute server path to static mount URL if needed,
              # or fetch directly via static server URL
              clip_filename = os.path.basename(clip_path)
              static_clip_url = f"{API_BASE_URL}/data/{clip_filename}"

              st.subheader("Generated Video Clip Preview")
              st.video(static_clip_url)
            else:
              st.warning("Match verified, but clip file path is missing.")
      except Exception as e:
        st.error(f"Failed to connect to backend API: {e}")
  else:
    st.warning("Please enter a search query.")