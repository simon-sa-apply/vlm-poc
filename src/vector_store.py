import hashlib
import chromadb
from src.config import CHROMA_PERSIST_PATH, CHROMA_COLLECTION_NAME, DEFAULT_N_RESULTS

def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_PATH)
    return client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

def generate_record_id(filename: str) -> str:
    return hashlib.sha256(filename.encode()).hexdigest()

def upsert_image(collection, record_id: str, embedding: list[float], category: str, path: str, filename: str):
    collection.upsert(
        ids=[record_id],
        embeddings=[embedding],
        metadatas=[{"category": category, "path": path}],
        documents=[filename]
    )

def query_similar(collection, query_embedding: list[float], n_results: int = DEFAULT_N_RESULTS) -> list[dict]:
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
    
    formatted_results = []
    if not results["ids"] or not results["ids"][0]:
        return formatted_results
        
    for i in range(len(results["ids"][0])):
        formatted_results.append({
            "rank": i + 1,
            "path": results["metadatas"][0][i]["path"],
            "category": results["metadatas"][0][i]["category"],
            "distance": results["distances"][0][i]
        })
    return formatted_results
