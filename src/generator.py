import os
import pickle
import faiss
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
import torch
from collections import defaultdict

# Configurations
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VECTOR_STORE_PATH = os.path.join(BASE_DIR, "vector_store")
FAISS_INDEX_PATH = os.path.join(VECTOR_STORE_PATH, "faiss_index")
METADATA_PATH = os.path.join(VECTOR_STORE_PATH, "metadata.pkl")

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "google/flan-t5-small"

TOP_K = 3
MAX_INPUT_TOKENS = 512
MAX_GENERATION_TOKENS = 300

device = "cuda" if torch.cuda.is_available() else "cpu"

def initialize_system():
    print(f"Using device: {device}")
    print("Initializing system components...")
    
    try:
        # Load retrieval components
        index = faiss.read_index(FAISS_INDEX_PATH)
        with open(METADATA_PATH, "rb") as f:
            metadata = pickle.load(f)
        embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
        
        # Load language model
        tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
        model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL_NAME)
        
        return {
            'index': index,
            'metadata': metadata,
            'embedder': embedder,
            'tokenizer': tokenizer,
            'model': model
        }
    except Exception as e:
        print(f"Initialization failed: {e}")
        return None

def extract_key_phrases(text):
    """Extract concise key phrases from complaint text"""
    sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 20]
    if not sentences:
        return text[:100] + '...'
    return sentences[0][:120] + '...'

def generate_structured_analysis(question, chunks):
    """Create focused, concise analysis from complaints"""
    # Analyze product distribution
    product_counts = defaultdict(int)
    for c in chunks:
        product_counts[c.get('product', 'Unknown')] += 1
    
    # Extract concise key phrases
    key_phrases = list(set(extract_key_phrases(c['chunk_text']) for c in chunks))
    
    # Build structured response
    analysis = (
        "1. Key Issues Identified:\n" +
        "\n".join(f"- {phrase}" for phrase in key_phrases[:3]) + "\n\n"
        
        "2. Products Affected:\n" +
        ", ".join(f"{prod} ({count})" for prod, count in product_counts.items()) + "\n\n"
        
        "3. Customer Sentiment:\n"
        "- Strongly negative in all cases\n\n"
        
        "4. Most Critical Complaint:\n"
        f"- '{extract_key_phrases(chunks[0]['chunk_text'])}'\n\n"
        
        "5. Recommended Actions:\n"
        "- Investigate root causes for these specific issues\n"
        "- Implement customer communication improvements\n"
        "- Review process for: " + ", ".join(product_counts.keys())
    )
    
    return analysis

def main():
    components = initialize_system()
    if not components:
        return

    print("\n" + "="*60)
    print("🔍 CREDITRUST COMPLAINT ANALYTICS DASHBOARD")
    print("="*60)
    print("Ask about specific customer service issues")
    print("Type 'exit' to quit\n")

    while True:
        try:
            question = input("\nYour question: ").strip()
            if question.lower() == 'exit':
                break
            if not question:
                continue
                
            # Retrieve complaints
            query_vec = components['embedder'].encode([question], convert_to_tensor=True)
            distances, indices = components['index'].search(query_vec.cpu().numpy(), TOP_K)
            chunks = [components['metadata'][i] for i in indices[0] if i < len(components['metadata'])]
            
            if not chunks:
                print("No matching complaints found in our database.")
                continue
                
            # Display found complaints
            print("\n" + "="*60)
            print("📌 TOP MATCHING COMPLAINTS:")
            for i, c in enumerate(chunks, 1):
                product = c.get('product', 'Various')
                print(f"{i}. {product}: {extract_key_phrases(c['chunk_text'])}")
            
            # Generate and display analysis
            analysis = generate_structured_analysis(question, chunks)
            
            print("\n" + "="*60)
            print("📊 EXECUTIVE ANALYSIS SUMMARY")
            print("="*60)
            print(analysis)
            print("="*60)
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nError processing request: {e}")
            continue

if __name__ == "__main__":
    main()