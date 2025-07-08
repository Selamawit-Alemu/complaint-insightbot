import os
import pickle
import faiss
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
import torch
from collections import defaultdict
from transformers import pipeline
# ==================== Configuration ====================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VECTOR_STORE_PATH = os.path.join(BASE_DIR, "vector_store")
FAISS_INDEX_PATH = os.path.join(VECTOR_STORE_PATH, "faiss_index")
METADATA_PATH = os.path.join(VECTOR_STORE_PATH, "metadata.pkl")

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
model1="google/flan-t5-small"
model2="google/flan-t5-base"
model3="google/flan-t5-large"
model="mistral-7b-instruct"
LLM_MODEL_NAME = model2

TOP_K = 5
MAX_GENERATION_TOKENS = 300

device = "cuda" if torch.cuda.is_available() else "cpu"

PROMPT_TEMPLATE = """
You are a financial analyst assistant for CrediTrust. Your task is to help internal teams understand customer pain points by analyzing complaint excerpts.

Use only the information from the context below to answer the question. If the context does not contain enough information, respond with "I don't have enough information."

Context:
{context}

Question:
{question}

Answer:
"""

# ==================== System Initialization ====================
def initialize_system():
    print(f"Using device: {device}")
    print("Initializing system components...")
    try:
        index = faiss.read_index(FAISS_INDEX_PATH)
        with open(METADATA_PATH, "rb") as f:
            metadata = pickle.load(f)
        embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
        tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
        model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL_NAME).to(device)

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

# ==================== Helper Functions ====================
def extract_key_phrases(text):
    sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 20]
    if not sentences:
        return text[:100] + '...'
    return sentences[0][:120] + '...'

def build_prompt(context_chunks, user_question):
    context = "\n---\n".join(c['chunk_text'] for c in context_chunks[:3])
    return PROMPT_TEMPLATE.format(context=context, question=user_question)


def generate_llm_answer(components, user_question, context_chunks):
    prompt = build_prompt(context_chunks, user_question)
    generator = pipeline("text2text-generation", model=LLM_MODEL_NAME, tokenizer=LLM_MODEL_NAME, device=0 if torch.cuda.is_available() else -1)

    response = generator(prompt, max_new_tokens=MAX_GENERATION_TOKENS, do_sample=True, top_p=0.9, temperature=0.7)
    return response[0]['generated_text']


def retrieve_top_k_chunks(components, question, k=TOP_K):
    query_vec = components['embedder'].encode([question], convert_to_tensor=True)
    distances, indices = components['index'].search(query_vec.cpu().numpy(), k)
    return [components['metadata'][i] for i in indices[0] if i < len(components['metadata'])]

# ==================== Main CLI ====================
def main():
    components = initialize_system()
    if not components:
        return

    print("\n" + "="*60)
    print("🔍 CREDITRUST COMPLAINT INSIGHTBOT")
    print("="*60)
    print("Ask about specific customer service issues.")
    print("Type 'exit' to quit.\n")

    while True:
        try:
            question = input("\nYour question: ").strip()
            if question.lower() == 'exit':
                break
            if not question:
                continue

            chunks = retrieve_top_k_chunks(components, question)

            if not chunks:
                print("No matching complaints found in our database.")
                continue

            print("\n" + "="*60)
            print("📌 TOP RETRIEVED COMPLAINTS:")
            for i, c in enumerate(chunks, 1):
                print(f"{i}. {extract_key_phrases(c['chunk_text'])}")

            print("\n" + "="*60)
            print("💡 LLM-GENERATED INSIGHT")
            print("="*60)
            print(generate_llm_answer(components, question, chunks))
            print("="*60)

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nError processing request: {e}")
            continue

if __name__ == "__main__":
    main()