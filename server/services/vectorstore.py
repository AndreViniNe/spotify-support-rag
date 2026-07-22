from pathlib import Path
from langchain_chroma import Chroma
import chromadb
from services.embeddings import load_embeddings

PATH = str(Path(__file__).resolve().parent.parent / "data" / "chroma")

def get_vector_store(embeddings_model: str = "sentence-transformers/all-MiniLM-L6-v2") -> Chroma:
    # Create a persistent Chroma vector store using the specified embeddings model and the defined path for storage.
    client = chromadb.PersistentClient(path=PATH)
    embeddings = load_embeddings(embeddings_model)
    return Chroma(
        client=client, 
        collection_name="spotify_support", 
        embedding_function=embeddings,
        persist_directory = PATH
    )