import os
import pickle
import faiss
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from sentence_transformers import SentenceTransformer
import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VECTOR_STORE_PATH = os.path.join(BASE_DIR, "vector_store")
FAISS_INDEX_PATH = os.path.join(VECTOR_STORE_PATH, "faiss_index")
METADATA_PATH = os.path.join(VECTOR_STORE_PATH, "metadata.pkl")

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "google/flan-t5-base"

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

def initialize_system():
    """
    Initialize and load the FAISS index, metadata, embedder, tokenizer, model, and text generation pipeline.
    """
    print(f"Using device: {device}")
    print("Initializing system components...")
    try:
        index = faiss.read_index(FAISS_INDEX_PATH)
        with open(METADATA_PATH, "rb") as f:
            metadata = pickle.load(f)

        embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)
        tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
        model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL_NAME).to(device)

        generator = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            device=0 if device == "cuda" else -1,
            max_length=MAX_GENERATION_TOKENS,
            do_sample=False,   # Set to False for more deterministic answers
            temperature=0.0    # Low temperature for factual responses
        )

        return {
            'index': index,
            'metadata': metadata,
            'embedder': embedder,
            'tokenizer': tokenizer,
            'model': model,
            'generator': generator
        }
    except Exception as e:
        print(f"Initialization failed: {e}")
        return None


def extract_key_phrases(text: str) -> str:
    """
    Extract a short summary snippet from the complaint chunk.
    """
    sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 20]
    if not sentences:
        return text[:100] + '...'
    return sentences[0][:120] + '...'


def build_prompt(context_chunks: list, user_question: str) -> str:
    """
    Build the prompt string to feed into the LLM by concatenating retrieved context chunks.
    """
    if not context_chunks:
        return PROMPT_TEMPLATE.format(context="No relevant information found.", question=user_question)
    context = "\n---\n".join(c['chunk_text'] for c in context_chunks[:3])
    return PROMPT_TEMPLATE.format(context=context, question=user_question)


def generate_llm_answer(components: dict, user_question: str, context_chunks: list) -> str:
    """
    Generate an answer from the LLM based on the question and retrieved context chunks.
    """
    prompt = build_prompt(context_chunks, user_question)
    if not context_chunks:
        # No relevant info - early return
        return "I don't have enough information."

    try:
        response = components['generator'](prompt)[0]['generated_text']
        return response.strip()
    except Exception as e:
        print(f"LLM generation error: {e}")
        return "Sorry, I could not generate an answer at this time."


def retrieve_top_k_chunks(components: dict, question: str, k: int = TOP_K) -> list:
    """
    Retrieve the top-k most relevant complaint chunks for a user question.
    """
    query_vec = components['embedder'].encode([question], convert_to_tensor=True)
    distances, indices = components['index'].search(query_vec.cpu().numpy(), k)

    # Filter indices out of range and get metadata
    results = []
    for i in indices[0]:
        if i < len(components['metadata']):
            results.append(components['metadata'][i])
    return results


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
            answer = generate_llm_answer(components, question, chunks)
            print(answer)
            print("="*60)

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nError processing request: {e}")
            continue


if __name__ == "__main__":
    main()
