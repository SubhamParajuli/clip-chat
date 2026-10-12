import hashlib
import os
import cv2
import easyocr
from sentence_transformers import SentenceTransformer
import torch
import whisper
from src.db_client import (
    get_audio_collection,
    get_chroma_client,
    get_visual_collection,
)
from yt_dlp import YoutubeDL

from src.logging_config import get_logger

logger = get_logger(__name__)


def compute_file_hash(file_path: str) -> str:
  """Computes an MD5 hash of a local file in chunks for quick,

  deterministic identification (rename-resilient).
  """
  hasher = hashlib.md5()
  with open(file_path, "rb") as f:
    for chunk in iter(lambda: f.read(4096 * 1024), b""):
      hasher.update(chunk)
  return f"vid_{hasher.hexdigest()[:12]}"


def ingest_video(
    video_path: str, video_id: str, similarity_threshold: float = 0.08
):
  if not os.path.exists(video_path):
    raise FileNotFoundError(f"Video file not found: {video_path}")

  client = get_chroma_client()
  audio_col = get_audio_collection(client)
  visual_col = get_visual_collection(client)

  # --- 1. CACHING & IDEMPOTENCY CHECK ---
  # If this video_id already exists in ChromaDB, skip encoding entirely!
  existing = audio_col.get(where={"video_id": video_id}, limit=1)
  if existing and existing.get("ids"):
    logger.info(
        "Cache hit for video ID %s; skipping redundant AI encoding.",
        video_id,
    )
    return
  # -------------------------------------

  logger.info("Cache miss; starting multi-modal ingestion for %s.", video_id)

  # Load models onto GPU if available
  device = "cuda" if torch.cuda.is_available() else "cpu"
  whisper_model = whisper.load_model("base", device=device)
  clip_model = SentenceTransformer("clip-ViT-B-32", device=device)
  text_embedder = SentenceTransformer("all-MiniLM-L6-v2", device=device)
  ocr_reader = easyocr.Reader(["en"], gpu=(device == "cuda"))

  # 2. INGEST AUDIO TRACK
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
            "file_path":video_path,
            "type": "audio",
            "start_time": float(seg["start"]),
            "end_time": float(seg["end"]),
        }],
    )
    audio_points += 1
  logger.info("Saved %d audio chunks.", audio_points)

  # 3. INGEST VISUAL FRAMES
  cap = cv2.VideoCapture(video_path)
  fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

  success, prev_frame = cap.read()
  if not success:
    cap.release()
    logger.error("Unable to read video frames from %s.", video_path)
    return

  prev_frame = cv2.resize(prev_frame, (640, 360))
  frame_count = 0
  saved_frames = 0
  skip_interval = int(fps)

  while success:
    success, frame = cap.read()
    if not success:
      break

    frame_count += 1
    if frame_count % skip_interval != 0:
      continue

    timestamp = frame_count / fps
    small_frame = cv2.resize(frame, (640, 360))

    diff = cv2.absdiff(small_frame, prev_frame)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    non_zero_count = cv2.countNonZero(gray_diff)
    total_pixels = small_frame.shape[0] * small_frame.shape[1]
    change_ratio = non_zero_count / total_pixels

    if change_ratio < similarity_threshold:
      continue

    prev_frame = small_frame
    ocr_results = ocr_reader.readtext(small_frame, detail=0)
    ocr_text = " ".join(ocr_results).strip()

    visual_vector = clip_model.encode(small_frame).tolist()
    doc_content = f"Visual frame at {timestamp:.1f}s."
    if ocr_text:
      doc_content += f" OCR text: {ocr_text}"

    metadata = {
        "video_id": video_id,
        "type": "visual",
        "file_path":video_path,
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
  logger.info(
      "Ingestion complete. Saved %d key visual frames out of %d total frames.",
      saved_frames,
      frame_count,
  )


def ingest_from_source(source: str, video_id: str = None) -> str:
  """Handles YouTube URLs or Local Paths, ensuring deterministic ID generation."""
  target_path = source
  resolved_video_id = video_id

  # Handle YouTube URLs
  if "youtube.com" in source or "youtu.be" in source:
    logger.info("Downloading YouTube video: %s", source)
    output_dir = "./data"
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
        resolved_video_id = f"yt_{info.get('id')}"

  # Handle Local Files / Direct Uploads
  else:
    if not os.path.exists(target_path):
      raise FileNotFoundError(f"Source file not found: {target_path}")

    # If no ID is provided, hash the file content to make it unique yet deterministic
    if not resolved_video_id:
      resolved_video_id = compute_file_hash(target_path)

  ingest_video(target_path, resolved_video_id)
  return resolved_video_id