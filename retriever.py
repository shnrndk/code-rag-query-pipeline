# retriever.py
from sentence_transformers import SentenceTransformer
import chromadb
import config

print("Loading retriever models...")
# Initialize these globally so they aren't re-loaded on every query
try:
    client = chromadb.PersistentClient(path=config.VECTOR_DB_PATH)
    collection = client.get_collection(config.COLLECTION_NAME)
    model = SentenceTransformer(config.EMBEDDING_MODEL)
except Exception as e:
    print(f"Error loading models or database. Did you run setup_db.py? Error: {e}")

def retrieve(query_text, top_k=config.TOP_K):
    """
    Embeds a query and returns the top k chunks from ChromaDB.
    """
    # 1. Embed the user's query
    query_emb = model.encode(query_text).tolist()
    
    # 2. Query the Chroma collection
    # Chroma uses cosine distance by default for all-MiniLM-L6-v2
    results = collection.query(
        query_embeddings=[query_emb],
        n_results=top_k
    )
    
    return results