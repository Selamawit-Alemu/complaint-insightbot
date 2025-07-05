import os
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pickle

# Define paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "filtered_complaints.csv")
VECTOR_STORE_PATH = os.path.join(BASE_DIR, "vector_store")
INDEX_PATH = os.path.join(VECTOR_STORE_PATH, "faiss_index")
METADATA_PATH = os.path.join(VECTOR_STORE_PATH, "metadata.pkl")

# Create vector_store directory
os.makedirs(VECTOR_STORE_PATH, exist_ok=True)

# Parameters
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def load_data(path):
    return pd.read_csv(path)

def chunk_texts(texts):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )
    return [splitter.split_text(text) for text in texts]

def flatten_chunks(df, chunked_texts):
    flattened = []
    for idx, chunks in enumerate(chunked_texts):
        complaint_id = df.iloc[idx]['complaint_id'] if 'complaint_id' in df.columns else idx
        product = df.iloc[idx]['product_mapped'] if 'product_mapped' in df.columns else None
        for i, chunk in enumerate(chunks):
            flattened.append({
                "chunk_id": f"{complaint_id}_{i}",
                "complaint_id": complaint_id,
                "product": product,
                "chunk_text": chunk   # ✅ include actual text
            })
    return flattened

def embed_texts(texts, model):
    return model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

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

    print("Embedding chunks...")
    texts = [item["chunk_text"] for item in flattened]
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    embeddings = embed_texts(texts, model)

    print("Building FAISS index...")
    index = build_faiss_index(embeddings, embeddings.shape[1])

    print("Saving index and metadata...")
    save_faiss_index(index, INDEX_PATH)
    save_metadata(flattened, METADATA_PATH)

    print("✅ Task 2 completed successfully.")

if __name__ == "__main__":
    main()
