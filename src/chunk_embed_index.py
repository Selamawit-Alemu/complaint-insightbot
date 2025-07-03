import os
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle




# print(os.path.exists("data/filtered_complaints.csv"))   # Should return True


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "filtered_complaints.csv")
VECTOR_STORE_PATH = os.path.join(os.path.dirname(__file__), '..', 'vector_store')
os.makedirs(VECTOR_STORE_PATH, exist_ok=True)

faiss_index_path = os.path.join(VECTOR_STORE_PATH, 'faiss_index')
METADATA_PATH = os.path.join(VECTOR_STORE_PATH, 'metadata.pkl')




# Parameters
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def load_data(path):
    df = pd.read_csv(path)
    return df

def chunk_texts(texts):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )
    all_chunks = []
    for text in texts:
        chunks = text_splitter.split_text(text)
        all_chunks.append(chunks)
    return all_chunks

def flatten_chunks(df, chunked_texts):
    # Flatten list of chunks with metadata for indexing
    flattened = []
    for idx, chunks in enumerate(chunked_texts):
        complaint_id = df.iloc[idx]['complaint_id'] if 'complaint_id' in df.columns else idx
        product = df.iloc[idx]['product_mapped'] if 'product_mapped' in df.columns else None
        for i, chunk in enumerate(chunks):
            flattened.append({
                "complaint_id": complaint_id,
                "product": product,
                "chunk_text": chunk,
                "chunk_id": f"{complaint_id}_{i}"
            })
    return flattened

def embed_texts(texts, model):
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    return embeddings

def build_faiss_index(embeddings, dimension):
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def save_faiss_index(index, path):
    faiss.write_index(index, path)

def save_metadata(metadata, path):
    with open(path, 'wb') as f:
        pickle.dump(metadata, f)

def main():
    print("Loading data...")
    df = load_data(DATA_PATH)

    print(f"Chunking {len(df)} complaint narratives...")
    chunked_texts = chunk_texts(df['cleaned_narrative'].tolist())

    print("Flattening chunks with metadata...")
    flattened = flatten_chunks(df, chunked_texts)

    chunk_texts_list = [item['chunk_text'] for item in flattened]

    print(f"Loading embedding model '{EMBEDDING_MODEL_NAME}'...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print("Embedding chunks...")
    embeddings = embed_texts(chunk_texts_list, model)

    print(f"Building FAISS index with dimension {embeddings.shape[1]}...")
    index = build_faiss_index(embeddings, embeddings.shape[1])

    # Prepare metadata without text (optional, keep chunk_id, complaint_id, product)
    metadata = [{
        "chunk_id": item["chunk_id"],
        "complaint_id": item["complaint_id"],
        "product": item["product"]
    } for item in flattened]

    print(f"Saving FAISS index to '{VECTOR_STORE_PATH}' and metadata to '{METADATA_PATH}'...")
    os.makedirs(os.path.dirname(VECTOR_STORE_PATH), exist_ok=True)
    save_faiss_index(index, VECTOR_STORE_PATH)
    save_metadata(metadata, METADATA_PATH)

    print("Task 2 completed successfully.")

if __name__ == "__main__":
    main()
