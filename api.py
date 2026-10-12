import os
from fastapi import FastAPI,HTTPException,UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from src.agent import VideoSearchResponse,query_video_agent_programmatic
from src.ingest import ingest_from_source,ingest_video
import asyncio
app=FastAPI(title="Multi-Modal Video Search Agent API.")

#Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

app.mount("/data",StaticFiles(directory="data"),name="data")

class IngestURLRequest(BaseModel):
    source:str
    video_id: str | None=None


@app.post("/ingest-url")
async def api_ingest_url(req:IngestURLRequest):
    """Ingest via Youtube URL or local server file path."""
    try:
        final_id=await asyncio.to_thread(
            ingest_from_source,req.source,req.video_id
        )
        return {"status":"success", "video_id":final_id}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload-video")
async def api_upload_video(
    file:UploadFile=File(...),
    video_id:str|None=Form(None)
):
    try:
        os.makedirs("./data",exist_ok=True)
        file_path=os.path.join("data",file.filename)

        with open(file_path,"wb")as f:
            content=await file.read()
            f.write(content)

        resolved_video_id = video_id or os.path.splitext(file.filename)[0]
        await asyncio.to_thread(ingest_video, file_path, resolved_video_id)
        
        return {
        "status": "success",
        "video_id": resolved_video_id,
        "file_path": file_path,
        }
    except Exception as e:
        raise HTTPException(status_code=500,detail=str(e))


@app.post("/search", response_model=VideoSearchResponse)
async def api_search(query: str):
  try:
    result = await query_video_agent_programmatic(query)
    return result
  except Exception as e:
    raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
  import uvicorn

  uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)