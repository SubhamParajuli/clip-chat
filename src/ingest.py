import os
import cv2
import easyocr
import whisper
from sentence_transformers import SentenceTransformer
from src.db_client import get_chroma_client,get_video_collection


def ingest_video(video_path:str,video_id:str):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_id}...")
    print("Initializzzing models or ingestion...")
    whisper_model=whisper.load_model("base")
    clip_model=SentenceTransformer("clip-ViT-B-32")
    text_embedder=SentenceTransformer("all-MiniLM-L6-v2")
    ocr_reader=easyocr.Reader(["en"])


    client=get_chroma_client()
    collection=get_video_collection(client)


    #Ingest audio track(spoken words)
    print("Transcribing audio track ....")
    audio_result=whisper_model.transcribe(video_path)

    for i,seg in enumerate(audio_result.get("segments",[])):
        text=seg["text"]
        vector= text_embedder.encode(text).tolist()
        collection.add(
            ids=[f"{video_id}_audio_{i}"],
            embeddings=[vector],
            metadatas=[{
                "video_id":video_id,
                "type":"audio",
                "start_time":float(seg["start"]),
                "end_time":float(seg["end"]),
            }],
        )

    # 2. Ingest Visual Frames & OCR (Objects, Text)  
    print("Processing visual frames and running ocr..")
    cap=cv2.VideoCapture(video_path)
    fps=cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count=0
    success,frame=cap.read()

    while success:
        timestamp=frame_count/fps

        if frame_count % int(fps) == 0:
            ocr_results=ocr_reader.readtext(frame,detail=0)
            ocr_text=" ".join(ocr_results)

            visual_vector=clip_model.encode(frame).tolist()
            doc_content = f"Visual frame at {timestamp:.1f}s. OCR text detected: {ocr_text}"
            collection.add(
                    ids=[f"{video_id}_frame_{frame_count}"],
                    embeddings=[visual_vector],
                    documents=[doc_content],
                    metadatas=[{
                        "video_id": video_id,
                        "type": "visual",
                        "start_time": float(timestamp),
                        "end_time": float(timestamp + 1.0),
                        "ocr_text": ocr_text,
                    }],
                )            
            success, frame = cap.read()
            frame_count += 1
    cap.release()
    print(f"Successfully ingested multi-modal data for video: {video_id}!")