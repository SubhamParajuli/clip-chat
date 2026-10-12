import asyncio
import os
import torch
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from src.agent import VideoSearchResponse, query_video_agent_programmatic
from src.ingest import ingest_from_source
from src.db_client import get_chroma_client
from src.utils import cleanup_stale_clips

@asynccontextmanager
async def lifespan(app: FastAPI):
  # Startup: Run cleanup task
  cleanup_stale_clips()
  yield
  # Shutdown: (Add any teardown logic here if needed)

app = FastAPI(title="Multi-Modal Video Search Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve generated clips statically for frontend previews
app.mount("/data", StaticFiles(directory="data"), name="data")


class IngestURLRequest(BaseModel):
  source: str
  video_id: str | None = None


@app.post("/ingest-url")
async def api_ingest_url(req: IngestURLRequest):
  """Ingest via YouTube URL or local file path."""
  try:
    final_id = await asyncio.to_thread(
        ingest_from_source, req.source, req.video_id
    )
    return {"status": "success", "video_id": final_id}
  except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload-video")
async def api_upload_video(
    file: UploadFile = File(...), video_id: str | None = Form(None)
):
  """Upload a video file directly via multipart/form-data."""
  try:
    os.makedirs("./data", exist_ok=True)
    file_path = os.path.join("./data", file.filename)

    with open(file_path, "wb") as buffer:
      content = await file.read()
      buffer.write(content)

    # Ingest using content-hashing pipeline
    final_id = await asyncio.to_thread(ingest_from_source, file_path, video_id)

    return {
        "status": "success",
        "video_id": final_id,
        "file_path": file_path,
    }
  except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))


@app.post("/search", response_model=VideoSearchResponse)
async def api_search(query: str):
  try:
    result = await query_video_agent_programmatic(query)
    return result
  except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
  try:
    client = get_chroma_client()
    # Ping ChromaDB to ensure connection is healthy
    client.heartbeat()
    return {
        "status": "healthy",
        "database": "connected",
        "gpu_available": torch.cuda.is_available(),
    }
  except Exception as e:
    raise HTTPException(
        status_code=503, detail=f"Service unhealthy: {str(e)}"
    )


if __name__ == "__main__":
  import uvicorn

  uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)