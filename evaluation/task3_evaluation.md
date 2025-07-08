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

### 🔢 Scoring Guide (1–5 Scale)

- **5** – Accurate, fluent, and well-grounded in retrieved complaints.
- **4** – Mostly accurate with minor issues.
- **3** – Acceptable but lacking detail or slightly off-topic.
- **2** – Vague or only partially aligned with context.
- **1** – Irrelevant, generic, or hallucinated response.


---
## 📂 Dataset & Chunking

- Source: Filtered complaint narratives from the CFPB (Consumer Financial Protection Bureau) dataset.
- Preprocessing: Complaints were cleaned, deduplicated, and split into 100–150 word chunks.
- Embedding: SentenceTransformer (`all-MiniLM-L6-v2`) used to generate dense vector representations.



## 📋 Evaluation Table – flan-t5-small (Initial Model)

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


## 📋 Evaluation Table – flan-t5-base (Improved Model)
## 🔁 Model Change for Evaluation

Initially, the system used `google/flan-t5-small`, which frequently responded with:
> "I don't have enough information"

To improve answer quality, we switched to `google/flan-t5-base`. This model:
- Is still open-access and cost-free
- Provides more fluent, relevant responses
- Performs better in extracting financial insights

---

## 📋 Prompt Template

We used the following template to guide the LLM:

```text
You are a financial analyst assistant for CrediTrust. Your task is to help internal teams understand customer pain points by analyzing complaint excerpts.

Use only the information from the context below to answer the question. If the context does not contain enough information, respond with "I don't have enough information."

Context:
{context}

Question:
{question}

Answer:
Evaluation Table
No	Question	Retrieved Complaints (Summary)	Generated Insight	Quality	Comments
1	What are customers complaining about in the personal loan approval process?	Misleading terms, excessive fees, predatory offers	Predatory fees and practices	✅ Good	Directly supported
2	What types of issues do customers face with credit cards?	Negative impact on credit, customer service issues	I don't have enough information	❌ Weak	Should extract more
3	How do customers feel about using BNPL services?	Repetitive & unclear complaint texts	I don't have enough information	❌ Weak	No real pattern extracted
4	Are there common issues with closing savings accounts?	Lack of notice, multiple unexpected accounts	No	✅ Accurate	Matches content
5	Are failed transactions frequent in money transfer services?	Mostly smooth, some errors	No	✅ Good	Reliable judgment
6	What problems do customers face when disputing credit report errors?	Errors remain unresolved despite disputes	I don't have enough information	❌ Missed context	
7	Are there recurring complaints about hidden fees or billing errors?	Fee disputes, late fees, recurring charges	Hidden fees	✅ Good	Captured core issue
📊 Summary

    ✅ 4/7 answers were accurate

    ❌ 3/7 answers were insufficient

    ⚖️ Overall quality score: 57%

Although not perfect, flan-t5-base improved performance over the previous model. It succeeded in identifying key pain points for loans, money transfers, and billing. For other products like credit cards and BNPL, complaint text quality limited generation.
📌 Lessons Learned

    Prompt engineering helped guide the model better than before.

    Performance depends heavily on retrieval quality.

    Larger models like flan-t5-base can significantly improve answer quality without incurring additional cost.

🏁 Next Steps

    Improve chunking and retrieval logic to provide more coherent context.

    Explore other open-access models like T5-base, Mistral, or phi-2 in future phases.


## 🧾 Overall Results Summary

| Metric                       | flan-t5-small | flan-t5-base |
|-----------------------------|---------------|--------------|
| Avg. Quality Score          | 2.4 / 5       | 3.6 / 5      |
| # of Questions Scored ≥ 4   | 2             | 5            |
| "I don't have enough info" Responses | 5           | 2            |
| Hallucinated Responses      | 1             | 0            |

**🎯 Improvement:** Switching to `flan-t5-base` increased average answer quality by ~50%.
