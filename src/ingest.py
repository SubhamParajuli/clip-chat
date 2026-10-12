import os
import cv2
import easyocr
import whisper
from sentence_transformers import SentenceTransformer
from src.db_client import get_chroma_client, get_video_collection


def ingest_video(video_path: str, video_id: str, similarity_threshold: float = 0.08):
  if not os.path.exists(video_path):
    raise FileNotFoundError(f"Video file not found: {video_path}")

  print(f"Starting optimized multi-modal ingestion for: {video_id}...")
  whisper_model = whisper.load_model("base")
  clip_model = SentenceTransformer("clip-ViT-B-32")
  text_embedder = SentenceTransformer("all-MiniLM-L6-v2")
  ocr_reader = easyocr.Reader(["en"])

  client = get_chroma_client()
  collection = get_video_collection(client)

  # 1. INGEST AUDIO TRACK (Whisper handles natural semantic chunking)
  print("Transcribing audio...")
  audio_result = whisper_model.transcribe(video_path)
  audio_points = 0
  for i, seg in enumerate(audio_result.get("segments", [])):
    text = seg["text"].strip()
    if not text:
      continue  # Skip empty audio segments

    vector = text_embedder.encode(text).tolist()
    collection.add(
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

  # 2. OPTIMIZED VISUAL INGESTION (Scene Change Detection + Conditional OCR)
  print("Processing visual frames with scene-change filtering...")
  cap = cv2.VideoCapture(video_path)
  fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
  
  success, prev_frame = cap.read()
  if not success:
    print("Error reading video frames.")
    cap.release()
    return

  frame_count = 0
  saved_frames = 専門店 = 0 # tracking saved frames

  while success:
    success, frame = cap.read()
    if not success:
      break
    
    frame_count += 1
    timestamp = frame_count / fps

    # SPACE SAVING TRICK 1: Skip frames that are virtually identical to the previous one
    diff = cv2.absdiff(frame, prev_frame)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    non_zero_count = cv2.countNonZero(gray_diff)
    total_pixels = frame.shape[0] * frame.shape[1]
    change_ratio = non_zero_count / total_pixels

    # If visual change is less than 8%, skip saving this frame entirely!
    if change_ratio < similarity_threshold and frame_count % int(fps) != 0:
      continue

    prev_frame = frame  # Update previous frame reference

    # SPACE SAVING TRICK 2: Run OCR only on dynamic frames
    ocr_results = ocr_reader.readtext(frame, detail=0)
    ocr_text = " ".join(ocr_results).strip()

    # Generate visual vector embedding via CLIP
    visual_vector = clip_model.encode(frame).tolist()
    doc_content = f"Visual frame at {timestamp:.1f}s."
    if ocr_text:
      doc_content += f" OCR text: {ocr_text}"

    # Build metadata (only include ocr_text if text was actually found to save JSON space)
    metadata = {
        "video_id": video_id,
        "type": "visual",
        "start_time": float(timestamp),
        "end_time": float(timestamp + 1.0),
    }
    if ocr_text:
      metadata["ocr_text"] = ocr_text

    collection.add(
        ids=[f"{video_id}_frame_{frame_count}"],
        embeddings=[visual_vector],
        documents=[doc_content],
        metadatas=[metadata],
    )
    saved_frames += 1

  cap.release()
  print(f"Optimized Ingestion Complete! Saved only {saved_frames} dynamic visual frames out of {frame_count} total frames.")