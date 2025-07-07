# 📊 Task 3: Qualitative Evaluation Report – Complaint InsightBot

## 🎯 Objective
To evaluate the effectiveness of the RAG (Retrieval-Augmented Generation) pipeline in answering user questions using customer complaint data. The focus is on how well the system retrieves relevant chunks and generates meaningful insights via the LLM.

---

## 🧪 Evaluation Setup

- **Retriever**: FAISS + `all-MiniLM-L6-v2`
- **LLM**: `flan-t5-small`
- **Prompt template**: Guides model to answer like a financial analyst using only retrieved context.
- **Top-K Chunks**: 5
- **Device**: CPU

---

## 📋 Evaluation Table

| # | Question | Generated Answer | Retrieved Sources | Score (1–5) | Comments |
|---|----------|------------------|-------------------|-------------|----------|
| 1 | What types of issues do customers face with CrediTrust credit cards? | "I don't have enough information" | Credit score harm, poor service, unhelpful | 2 | Answer vague, context existed |
| 2 | What are customers complaining about in the personal loan approval process? | Excessive fees, predatory lending | Yes – multiple loan issues | 4.5 | Solid answer, well supported |
| 3 | How do customers feel about using BNPL services from CrediTrust? | "I don't have enough information" | Repetitive “open-end credit” chunks | 1 | Chunk quality poor |
| 4 | Are there common issues when customers try to close savings accounts? | "No" | Closure policy & miscommunication | 3.5 | Acceptable, could be more nuanced |
| 5 | Are failed transactions a frequent issue in money transfer services? | "No" | Mixed reviews, some errors | 4 | Reliable, concise |
| 6 | Are there patterns of unauthorized charges or fraud? | "unauthorized charges" | Disputes, fraud scheme patterns | 4 | Acceptable summary |
| 7 | How do customers describe their experience with support? | "I don't have enough information" | Long-term issues, rude support | 2 | Should be more expressive |
| 8 | What problems do customers face disputing credit report errors? | "I don't have enough information" | Inaccuracies, repeated disputes | 2.5 | Slightly underperforms |
| 9 | Are there recurring complaints about billing errors or fee transparency? | "Thank you" + irrelevant billing | Late fees, transparency gaps | 2 | LLM misinterpreted prompt |
| 10 | What are the most common pain points across all CrediTrust products? | "I don't have enough information" | Many strong complaints retrieved | 1.5 | Missed summarization |

---

## 🔍 Insights & Recommendations

### ✅ Strengths
- FAISS retriever works well with Sentence-BERT
- Strong performance for loan-related questions and fraud-related complaints
- Effective modularity and CLI usability

### ❌ Weaknesses
- LLM (`flan-t5-small`) too limited for complex summarization
- Prompt needs improvement for better summarization and tone
- Some complaints are repetitive or noisy — needs chunk cleaning

### 🔧 Next Steps
- Upgrade to a stronger model (e.g., `flan-t5-base`, `mistral-7b`, or `gpt-3.5-turbo`)
- Preprocess chunks more aggressively (deduplication, sentence completion)
- Refine prompt with few-shot examples or role reinforcement

---

## ✅ Conclusion
While the system handles some questions well, particularly around loans and money transfers, many responses lack depth. Improving model strength and prompt design will significantly enhance performance.

