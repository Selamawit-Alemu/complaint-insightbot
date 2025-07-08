import streamlit as st
import time
from typing import List, Dict, Generator

# ---------------------- Mocked Backend Logic ---------------------- #

def initialize_system() -> Dict:
    """
    Mock system initialization. Replace with real init logic.
    """
    st.info("✅ System initialized.")
    return {"status": "initialized", "llm_client": "mock_llm"}

def retrieve_top_k_chunks(components: Dict, query: str) -> List[Dict]:
    """
    Mock retrieval function. Replace with real chunk retrieval logic.
    """
    query_lower = query.lower()

    if "loan" in query_lower or "interest" in query_lower:
        return [
            {"chunk_text": "Complaint about predatory interest rates on personal loans."},
            {"chunk_text": "Confusion about credit card interest fees."},
            {"chunk_text": "Loan disbursement delay during urgent need."}
        ]
    elif "fraud" in query_lower:
        return [
            {"chunk_text": "Unauthorized credit card charges reported."},
            {"chunk_text": "Fraud claim not resolved for over 2 weeks."}
        ]
    elif "credit card" in query_lower:
        return [
            {"chunk_text": "Unexpected annual fees and chargeback issues on credit card."}
        ]
    else:
        return []

def generate_llm_answer_stream(components: Dict, user_input: str, chunks: List[Dict]) -> Generator[str, None, None]:
    """
    Mock streamed generation. Replace with actual LLM streaming if needed.
    """
    mock_answer = "Based on retrieved complaints, customers often experience unclear terms and delayed resolutions."
    for word in mock_answer.split():
        yield word + " "
        time.sleep(0.03)

def extract_key_phrases(text: str) -> str:
    return text[:100] + "..." if len(text) > 100 else text

# ---------------------- Streamlit UI ---------------------- #

def render_header():
    st.markdown("""
    <h1 style='position: fixed; top: 0; left: 0; width: 100%; padding: 0.5rem 15rem;
    background-color: black; color: white; z-index: 999;'>🔍 CrediTrust Complaint InsightBot</h1>
    <div style='height: 4rem'></div>
    """, unsafe_allow_html=True)
    st.markdown("Ask about customer issues across products like credit cards, loans, and transfers.")

def render_messages():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def render_sources(chunks: List[Dict]):
    if chunks:
        st.markdown("### 📌 Top Retrieved Complaints:")
        for i, chunk in enumerate(chunks, 1):
            snippet = extract_key_phrases(chunk.get("chunk_text", ""))
            st.markdown(f"- **Complaint {i}:** {snippet}")

# ---------------------- App Logic ---------------------- #

def handle_submission():
    query = st.session_state.current_input.strip()
    if not query:
        return

    st.session_state.messages.append({"role": "user", "content": query})
    st.session_state.submitted = True
    st.session_state.current_input = ""  # Clear input after submission

def main():
    st.set_page_config(page_title="CrediTrust Complaint InsightBot", layout="centered")

    # ---- State Initialization ----
    if "components" not in st.session_state:
        st.session_state.components = initialize_system()
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "current_input" not in st.session_state:
        st.session_state.current_input = ""
    if "submitted" not in st.session_state:
        st.session_state.submitted = False

    render_header()
    render_messages()

    # ---- Input & Buttons ----
    user_input = st.text_input(
        "💬 Enter your question:",
        value=st.session_state.current_input,
        key="user_input_text_box",
        on_change=lambda: st.session_state.update(current_input=st.session_state.user_input_text_box)
    )

    col1, col2 = st.columns([1, 1])
    ask_disabled = not st.session_state.current_input.strip()
    submit_button = col1.button("Ask", disabled=ask_disabled, on_click=handle_submission)
    clear_button = col2.button("Clear", on_click=lambda: st.session_state.update(messages=[], current_input="", submitted=False))

    # ---- Handle Submission ----
    if st.session_state.submitted:
        st.session_state.submitted = False  # reset flag after handling

        query = st.session_state.messages[-1]["content"]
        components = st.session_state.components

        with st.spinner("🔍 Analyzing complaints..."):
            try:
                chunks = retrieve_top_k_chunks(components, query)

                if not chunks:
                    assistant_reply = (
                        "❌ No relevant complaints found.\n\n"
                        "Try asking about something more specific, like:\n"
                        "- 'Common issues with loan disbursement?'\n"
                        "- 'Any fraud reports on money transfers?'"
                    )
                else:
                    answer_placeholder = st.empty()
                    full_answer = ""
                    for piece in generate_llm_answer_stream(components, query, chunks):
                        full_answer += piece
                        answer_placeholder.markdown(full_answer)

                    render_sources(chunks)
                    assistant_reply = f"## 💡 AI Insight\n\n{full_answer}"

                st.session_state.messages.append({"role": "assistant", "content": assistant_reply})

            except Exception as e:
                error_msg = f"❗ Error: {e}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

if __name__ == "__main__":
    main()
