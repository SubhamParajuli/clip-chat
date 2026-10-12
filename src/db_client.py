import chromadb
import os

def get_chroma_client(path:str="../db"):
    """Returns a persistent local ChromaDb client"""
    if not os.path.exists("../db"):
        os.makedirs("../db",exist_ok=True)

    return chromadb.PersistentClient(path=path)

def get_video_collection(client:chromadb.PersistentClient,name:str="multi_model_video_index"):
    """Retrieves or creates the vector collection for video data."""  
    return client.get_or_create_collection(name=name)  

    