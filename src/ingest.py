import os
import cv2
import easyocr
import whisper
from sentence_transformers import SentenceTransformer
from yt_dlp import YoutubeDL
from src.db_client import (
    get_audio_collection,
    get_chroma_client,
    get_visual_collection,
)
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using compute device: {DEVICE}")


def ingest_video(
    video_path: str, video_id: str, similarity_threshold: float = 0.08
):
  if not os.path.exists(video_path):
    raise FileNotFoundError(f"Video file not found: {video_path}")
  existing = audio_col.get(where={"video_id": video_id}, limit=1)

  print(f"Starting optimized multi-modal ingestion for: {video_id}...")
  whisper_model = whisper.load_model("base",device=DEVICE)
  clip_model = SentenceTransformer("clip-ViT-B-32",device=DEVICE)
  text_embedder = SentenceTransformer("all-MiniLM-L6-v2",device=DEVICE)
  ocr_reader = easyocr.Reader(["en"],gpu=True)

  client = get_chroma_client()
  audio_col = get_audio_collection(client)
  visual_col = get_visual_collection(client)
  if existing["ids"]:
    print(f"Video '{video_id}' is already ingested. Skipping.")
    return
  # 1. INGEST AUDIO TRACK (384-dim)
  print("Transcribing audio...")
  audio_result = whisper_model.transcribe(video_path)
  audio_points = 0
  for i, seg in enumerate(audio_result.get("segments", [])):
    text = seg["text"].strip()
    if not text:
      continue

    vector = text_embedder.encode(text).tolist()
    audio_col.add(
        ids=[f"{video_id}_audio_{i}"],
        embeddings=[vector],
        documents=[text],
        metadatas=[{
            "video_id": video_id,
            "type": "audio",
            "start_time": float(seg["start"]),
            "end_time": float(seg["end"]),
        }],
    )
    audio_points += 1
  print(f"Saved {audio_points} audio chunks.")

  # 2. OPTIMIZED VISUAL INGESTION (Resized frames + Scene-Change + Fast OCR)
  print(
      "Processing visual frames with scene-change filtering & downscaling..."
  )
  cap = cv2.VideoCapture(video_path)
  fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

  success, prev_frame = cap.read()
  if not success:
    print("Error reading video frames.")
    cap.release()
    return

  # Resize previous frame reference for fast diff calculation
  prev_frame = cv2.resize(prev_frame, (640, 360))

  frame_count = 0
  saved_frames = 0
  skip_interval = int(fps)  # Sample roughly 1 frame per second by default

  while success:
    success, frame = cap.read()
    if not success:
      break

    frame_count += 1

    # Skip frames to target ~1 frame per second check unless heavy motion occurs
    if frame_count % skip_interval != 0:
      continue

    timestamp = frame_count / fps

    # Downscale frame to 640x360 for lightning-fast CPU inference
    small_frame = cv2.resize(frame, (640, 360))

    # Fast pixel difference check
    diff = cv2.absdiff(small_frame, prev_frame)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    non_zero_count = cv2.countNonZero(gray_diff)
    total_pixels = small_frame.shape[0] * small_frame.shape[1]
    change_ratio = non_zero_count / total_pixels

    # If change is minor, skip heavy OCR and CLIP embedding
    if change_ratio < similarity_threshold:
      continue

    prev_frame = small_frame

    # Run EasyOCR and CLIP on the downscaled frame (much faster on CPU)
    ocr_results = ocr_reader.readtext(small_frame, detail=0)
    ocr_text = " ".join(ocr_results).strip()

    visual_vector = clip_model.encode(small_frame).tolist()
    doc_content = f"Visual frame at {timestamp:.1f}s."
    if ocr_text:
      doc_content += f" OCR text: {ocr_text}"

    metadata = {
        "video_id": video_id,
        "type": "visual",
        "start_time": float(timestamp),
        "end_time": float(timestamp + 1.0),
    }
    if ocr_text:
      metadata["ocr_text"] = ocr_text

    visual_col.add(
        ids=[f"{video_id}_frame_{frame_count}"],
        embeddings=[visual_vector],
        documents=[doc_content],
        metadatas=[metadata],
    )
    saved_frames += 1

  cap.release()
  print(
      f"Ingestion Complete! Saved {saved_frames} key visual frames out of"
      f" {frame_count} total frames."
  )


def ingest_from_source(source: str, video_id: str = None) -> str:
  target_path = source
  resolved_video_id = video_id

  if "youtube.com" in source or "youtu.be" in source:
    print(f"Detected YouTube link. Downloading via yt-dlp: {source}")
    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)

    ydl_opts = {
        "format": "bv*+ba/b",
        "outtmpl": os.path.join(output_dir, "%(id)s.%(ext)s"),
        "quiet": False,
        "merge_output_format": "mp4",
    }

    with YoutubeDL(ydl_opts) as ydl:
      info = ydl.extract_info(source, download=True)
      target_path = ydl.prepare_filename(info)
      if not os.path.exists(target_path):
        base, _ = os.path.splitext(target_path)
        if os.path.exists(base + ".mp4"):
          target_path = base + ".mp4"
      if not resolved_video_id:
        resolved_video_id = info.get("id")

  if not resolved_video_id:
    resolved_video_id = os.path.splitext(os.path.basename(target_path))[0]

  ingest_video(target_path, resolved_video_id)
  return resolved_video_id