import chromadb
from chromadb.config import Settings

def get_chroma_client():
  # Local persistent client
  return chromadb.PersistentClient(path="data/chroma_db")


def get_audio_collection(client):
  return client.get_or_create_collection(
      name="video_audio_embeddings", metadata={"hnsw:space": "cosine"}
  )


def get_visual_collection(client):
  return client.get_or_create_collection(
      name="video_visual_embeddings", metadata={"hnsw:space": "cosine"}
  )