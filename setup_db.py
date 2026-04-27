# setup_db.py
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import chromadb
import config

def populate_database():
    print("Loading CodeSearchNet dataset (Python split)...")
    # Using trust_remote_code=True as required by some HF datasets [cite: 82]
    ds = load_dataset('code_search_net', 'python', split='train', streaming=True, trust_remote_code=True)
    
    print(f"Loading embedding model: {config.EMBEDDING_MODEL}...")
    model = SentenceTransformer(config.EMBEDDING_MODEL)
    
    print("Initializing ChromaDB client...")
    client = chromadb.PersistentClient(path=config.VECTOR_DB_PATH)
    col = client.get_or_create_collection(config.COLLECTION_NAME)
    
    items = []
    print("Fetching 1,000 functions...")
    for i, row in enumerate(ds):
        items.append(row)
        if len(items) >= 1000: 
            break
            
    # Combine docstring and code for richer embeddings [cite: 90]
    texts = [r['func_documentation_string'] + '\n' + r['func_code_string'] for r in items]
    
    print("Embedding texts... (This will be very fast on your machine)")
    embs = model.encode(texts, show_progress_bar=True).tolist()
    
    print("Adding records to ChromaDB...")
    col.add(
        ids=[str(i) for i in range(len(items))],
        embeddings=embs,
        documents=texts,
        metadatas=[{
            'func_name': r['func_name'],
            'repo': r['repository_name'],
            'path': r['func_path_in_repository']
        } for r in items]
    )
    print(f"Success! Database persisted to {config.VECTOR_DB_PATH}")

if __name__ == "__main__":
    populate_database()