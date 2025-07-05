import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VECTOR_STORE_DIR = os.path.join(BASE_DIR, "vector_store")
INDEX_PATH = os.path.join(VECTOR_STORE_DIR, "faiss_index")
METADATA_PATH = os.path.join(VECTOR_STORE_DIR, "metadata.pkl")

def load_vector_store(index_path, metadata_path):
    index = faiss.read_index(index_path)
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    return index, metadata

def embed_query(query, model):
    return model.encode([query], convert_to_numpy=True)[0]

def retrieve_top_k(query, index, metadata, model, k=5):
    query_vec = embed_query(query, model)
    D, I = index.search(np.array([query_vec]), k)
    results = []
    for i in I[0]:
        meta = metadata[i]
        results.append(meta)
    return results

def main():
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    index, metadata = load_vector_store(INDEX_PATH, METADATA_PATH)

    query = input("Enter your question: ")
    retrieved = retrieve_top_k(query, index, metadata, model)

    print("\nTop relevant chunks retrieved:")
    for i, item in enumerate(retrieved, 1):
        print(f"{i}. Complaint ID: {item['complaint_id']}, Product: {item['product']}")
        print(f"   Text: {item['chunk_text'][:300]}...\n")  # Only show first 300 characters

if __name__ == "__main__":
    main()
