# app.py
import streamlit as st
import time  # For simulating streaming delay

# --- MOCKING generator functions for demonstration purposes ---

def initialize_system():
    st.info("✅ System initialized.")
    return {"status": "initialized", "llm_client": "mock_llm"}

def retrieve_top_k_chunks(components, query):
    query_lower = query.lower()
    if query_lower in ["hi", "hello", "hey"]:
        return [{"chunk_text": "Hello! Ask me about customer complaints regarding credit cards, loans, or transfers."}]

    if "loan" in query_lower or "interest" in query_lower:
        return [
            {"chunk_text": "Customer complained about predatory interest rates on their personal loan, stating they felt misled by the initial offer."},
            {"chunk_text": "The terms and conditions for the credit card were not clearly explained, leading to unexpected fees."},
            {"chunk_text": "A user reported difficulty understanding the repayment schedule for their home loan, causing confusion and stress."},
            {"chunk_text": "Unclear eligibility criteria caused multiple loan application rejections."},
            {"chunk_text": "Delayed loan disbursement created financial stress for applicants."}
        ]
    elif "fraud" in query_lower or "unauthorized" in query_lower:
        return [
            {"chunk_text": "Customers reported multiple unauthorized charges on their credit cards."},
            {"chunk_text": "Delayed fraud resolution left users dissatisfied."},
            {"chunk_text": "Fraud department was hard to reach, and communication was inconsistent."}
        ]
    elif "close" in query_lower or "closure" in query_lower or "account" in query_lower:
        return [
            {"chunk_text": "Users faced hidden fees during account closure."},
            {"chunk_text": "Account closures were processed without proper notifications."},
            {"chunk_text": "Some customers reported account closure requests being ignored or delayed."}
        ]
    elif "transfer" in query_lower or "money transfer" in query_lower:
        return [
            {"chunk_text": "Transfers were delayed for multiple users, especially during weekends."},
            {"chunk_text": "Funds were deducted but not received by the recipient, causing concern."},
            {"chunk_text": "Poor customer service response to transfer issues caused frustration."}
        ]
    elif "credit card" in query_lower:
        return [
            {"chunk_text": "Annual fees for credit cards were not disclosed upfront."},
            {"chunk_text": "Users faced difficulty disputing charges on their CrediTrust credit card."},
            {"chunk_text": "Unexpected interest charges despite timely payments."}
        ]
    elif "pain point" in query_lower or "problem" in query_lower or "issue" in query_lower:
        return [
            {"chunk_text": "Loan approval delays and unclear terms are frequent issues."},
            {"chunk_text": "Customer service response time and fraud resolution also frustrate users."},
            {"chunk_text": "Platform navigation and digital access inconsistencies cause confusion."}
        ]
    else:
        return []

def generate_llm_answer_stream(components, user_input, chunks):
    joined_chunks = " ".join([c["chunk_text"].lower() for c in chunks])

    # Custom handling for greetings
    if any(greet in user_input.lower() for greet in ["hi", "hello", "hey"]):
        summary_topic = (
            "**Insight:** Hi there! I'm your assistant for exploring customer complaints. "
            "Ask me about issues like credit card disputes, loan delays, fraud reports, and more."
        )
    elif "fraud" in joined_chunks or "unauthorized" in joined_chunks:
        summary_topic = (
            "**Insight:** Based on the retrieved complaints, customers are frustrated with delayed fraud resolution "
            "and multiple unauthorized charges. Timely and transparent fraud handling processes would improve trust and satisfaction."
        )
    elif "close" in joined_chunks or "closure" in joined_chunks:
        summary_topic = (
            "**Insight:** Customers express concerns about hidden fees and poor communication during account closures. "
            "Ensuring clear procedures and timely updates could reduce dissatisfaction."
        )
    elif "loan" in joined_chunks or "interest" in joined_chunks:
        summary_topic = (
            "**Insight:** Customers are concerned about unclear loan terms, high interest rates, and loan disbursement delays. "
            "Transparency in product disclosures and streamlined application processes are key improvement areas."
        )
    elif "transfer" in joined_chunks:
        summary_topic = (
            "**Insight:** Customers frequently report delays and failures in money transfers. Improving backend processing times "
            "and better customer support during transaction failures can boost user confidence."
        )
    elif "credit card" in joined_chunks:
        summary_topic = (
            "**Insight:** Common issues with credit cards include hidden fees, unexpected charges, and dispute resolution difficulties. "
            "Greater fee transparency and robust support channels can enhance user trust."
        )
    else:
        summary_topic = (
            "**Insight:** Based on provided complaints, users experience diverse challenges across products. Clear communication, responsive support, "
            "and fair financial practices are consistent areas for improvement."
        )

    for word in summary_topic.split(" "):
        yield word + " "
        time.sleep(0.03)


def extract_key_phrases(text):
    return text[:100] + "..." if len(text) > 100 else text

# --- UI Initialization ---

if "components" not in st.session_state:
    st.session_state["components"] = initialize_system()
components = st.session_state["components"]

st.set_page_config(page_title="CrediTrust Complaint InsightBot", layout="centered")
st.markdown("""
    <h1 style='position: fixed; top: 0; left: 0; width: 100%; padding: 0.5rem 15rem; background-color: black; border-bottom: 1px solid #ddd; z-index: 999;'>🔍 CrediTrust Complaint InsightBot</h1>
    <div style='height: 4rem'></div>
""", unsafe_allow_html=True)

st.markdown("Ask about customer issues across products like credit cards, loans, and transfers.")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_input" not in st.session_state:
    st.session_state.current_input = ""

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input_placeholder = st.empty()
user_input_text = user_input_placeholder.text_input(
    "💬 Enter your question:",
    value=st.session_state.current_input,
    key="user_input_text_box",
    on_change=lambda: st.session_state.update(current_input=st.session_state.user_input_text_box)
)

col1, col2 = st.columns([1, 1])
submit_button = col1.button("Ask", key="ask_button")
clear_button = col2.button("Clear", key="clear_button")

if clear_button:
    st.session_state.messages = []
    st.session_state.current_input = ""
    st.rerun()

if submit_button or (user_input_text and st.session_state.current_input != user_input_text):
    query = st.session_state.current_input.strip()

    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        st.session_state.current_input = ""
        user_input_placeholder.text_input("💬 Enter your question:", value="", key="user_input_text_box_cleared")

        if not components:
            error_msg = "System failed to initialize."
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
        else:
            with st.spinner("🔍 Analyzing complaints..."):
                try:
                    chunks = retrieve_top_k_chunks(components, query)
                    ai_response_content = ""

                    if not chunks:
                        ai_response_content = ("❌ No relevant complaints found for that question.\n\n"
                                               "👉 Try rephrasing or ask about a specific issue like:\n"
                                               "- 'What are common complaints about credit card billing?'\n"
                                               "- 'Are there issues with loan approval delays?'")
                        with st.chat_message("assistant"):
                            st.warning(ai_response_content)
                    else:
                        sources_markdown = "### 📌 Top Retrieved Complaints:\n"
                        for i, c in enumerate(chunks, 1):
                            sources_markdown += f"- **Complaint {i}:** {extract_key_phrases(c['chunk_text'])}\n"

                        with st.chat_message("assistant"):
                            st.markdown("## 💡 **AI Insight**")
                            answer_placeholder = st.empty()
                            full_answer = ""
                            for chunk in generate_llm_answer_stream(components, query, chunks):
                                full_answer += chunk
                                answer_placeholder.markdown(full_answer)

                            st.markdown(sources_markdown)
                            ai_response_content = f"## 💡 **AI Insight**\n{full_answer}\n\n{sources_markdown}"

                    st.session_state.messages.append({"role": "assistant", "content": ai_response_content})

                except Exception as e:
                    error_message = f"❗ Error: {e}"
                    st.error(error_message)
                    st.session_state.messages.append({"role": "assistant", "content": error_message})

    st.rerun()