🧠 COMPLAINT-INSIGHTBOT

Complaint InsightBot is an internal Retrieval-Augmented Generation (RAG) tool built for CrediTrust Financial to transform large volumes of customer complaint narratives into strategic product insights. This system empowers internal teams (Product, Compliance, Support) to ask plain-English questions and receive synthesized, evidence-backed answers from 5M+ complaint records.

🚀 Project Goal

To reduce the time it takes internal teams to understand customer pain points — especially in high-volume services like Credit Cards, BNPL, Savings, Loans, and Money Transfers — by using AI-powered semantic search and summarization on raw customer complaint data.

📁 Repository Structure

        complaint-insightbot/
        │
        ├── .github/workflows/          # CI pipelines
        │   └── ci.yml
        ├── data/                       # Raw and filtered dataset
        │   ├── complaints.csv
        │   └── filtered_complaints.csv
        ├── notebooks/                  # Development notebooks
        │   ├── eda.ipynb
        │   └── data_chunking.ipynb
        ├── src/                        # Core logic for chunking, embedding, retrieval, and generation
        │   ├── __init__.py
        │   ├── chunk_embed_index.py
        │   ├── retriever.py
        │   └── generator.py
        ├── vector_store/               # Persisted FAISS index and metadata (excluded from GitHub)
        │   ├── faiss_index
        │   └── metadata.pkl
        ├── venv/                       # Virtual environment (excluded)
        ├── .gitignore
        ├── README.md                   # This file
        └── requirements.txt            # Python dependencies

    🔍 Note: The EDA notebook (eda.ipynb) includes narrative length analysis. Additional visualizations will be expanded in the final version.

## 📊 Exploratory Data Analysis (EDA) Summary

To gain a comprehensive understanding of complaint trends and data quality, we analyzed a 5 million-row sample from the CFPB complaints dataset. Among these, approximately 1.42 million entries contained a valid consumer complaint narrative, while over 3.57 million records lacked narrative data and were excluded from downstream tasks. Narrative lengths varied widely, ranging from very short (under 10 words) to over 6,000 words, with most complaints falling between 100–250 words—offering an ideal balance of context and brevity for embedding and summarization.

The product distribution showed a significant imbalance, with credit-related services dominating the dataset. Key categories included Credit Reporting (around 3.4 million complaints), Debt Collection (~274K), Checking/Savings (~114K), Credit Cards (~87K), and Money Transfers (~84K). For project focus, we filtered complaints to five main products: Credit Cards, Personal Loans, Buy Now, Pay Later (BNPL), Savings Accounts, and Money Transfers. After filtering and rigorous preprocessing—such as removing null narratives, cleaning noisy text, and normalizing content—we curated a high-quality dataset suitable for semantic retrieval.

This cleaned and filtered dataset serves as the foundation for downstream tasks, including text chunking, embedding with sentence-transformers, and vector indexing via FAISS. It enables efficient, context-rich retrieval for the RAG pipeline to generate actionable insights, helping internal stakeholders quickly identify and address key customer pain points.
## Chunking Strategy and Embedding Model Choice
# Chunking Strategy
🔹 Chunking Strategy

Processing large complaint narratives as a single unit reduces semantic retrieval quality and may exceed model input limits. To address this, we implemented a robust chunking strategy using LangChain’s RecursiveCharacterTextSplitter:

    Chunk Size: 500 characters
    This size was selected after experimentation, striking a balance between context preservation and compactness. It ensures each chunk carries sufficient meaning for accurate embeddings.

    Chunk Overlap: 50 characters
    This overlap helps maintain semantic continuity between chunks, reducing the risk of losing critical information at chunk boundaries.

    Separators Used:
    ["\n\n", "\n", ".", "!", "?", ",", " "]
    These hierarchical split points ensure that text is divided at natural linguistic boundaries, resulting in cleaner and more meaningful chunks.

    Scaling Consideration:
    Due to local compute constraints, chunking and preprocessing were offloaded to Google Colab with GPU support, where a subset of 1.4M complaint narratives was processed from a total of 5M records.

This method prepares complaint texts for effective embedding while preserving semantic granularity, which significantly boosts retrieval accuracy in downstream QA.
🔹 Embedding Model Choice

We adopted the sentence-transformers/all-MiniLM-L6-v2 model as the embedding backbone for the RAG system:

    ✅ Semantic Accuracy:
    It produces dense 384-dimensional vectors that capture sentence-level semantic similarity with high fidelity—essential for relevant chunk retrieval.

    ⚡ Lightweight and Fast:
    Its small size makes it computationally efficient for batch embedding over millions of chunks without overwhelming memory or disk resources.

    🧠 Plug-and-Play with FAISS & LangChain:
    The model integrates seamlessly into Python NLP pipelines and supports both CPU and GPU execution, aligning well with both local and Colab-based workflows.

🗃️ Embedding & Indexing

    Each chunk was embedded and stored using FAISS as the vector store backend.

    Metadata saved per chunk:

        Product

        Original Complaint Text

        Complaint ID
        This metadata is essential for contextual traceability and interpretability during semantic search and answer generation.

    The final vector store is saved in the vector_store/ directory. It contains:

        faiss_index (FAISS binary)

        metadata.pkl (original complaint metadata)
✂️ Text Chunking, Embedding, and Vector Indexing

    Used LangChain's RecursiveCharacterTextSplitter to break narratives into 500-character chunks with 50-character overlap.

    Generated dense embeddings using sentence-transformers/all-MiniLM-L6-v2.

    Stored embeddings in FAISS for fast vector similarity search with complaint metadata.

    Script: src/chunk_embed_index.py

    🧑‍💻 If you're using Google Colab, please run notebooks/data_chunking.ipynb instead of the .py script for chunking and embedding. It is optimized to run with GPU acceleration and large dataset support.

🧪 QA Generator CLI (Demo)

src/generator.py is a command-line tool that allows internal users to ask natural language questions like:

    "Why are customers unhappy with their savings accounts?"

It retrieves top-matching complaint chunks, summarizes key phrases, sentiment, product mentions, and recommends actions.

    ✅ Uses Google’s flan-t5-small for answer generation.

🛠️ Setup Instructions

    Create virtual environment & activate:

        python -m venv venv
        source venv/bin/activate  # On Windows: venv\Scripts\activate

    Install dependencies:

        pip install -r requirements.txt

    Run chunking & indexing (optional if vector store exists):

        python src/chunk_embed_index.py

    Launch the generator CLI:

        python src/generator.py

✅ Interim Deliverables Summary

    ✅ Cleaned and filtered dataset with valid complaint narratives

    ✅ EDA notebook with narrative statistics

    ✅ Semantic chunking strategy using LangChain

    ✅ Embedding pipeline using SentenceTransformers + FAISS

    ✅ Generator CLI for interactive question answering

    ✅ CI workflow using GitHub Actions (.github/workflows/ci.yml)

🔮 Next Steps

    Add multi-product filters to the retriever

    Develop Streamlit UI for internal users

    Tune prompt templates for better summarization

    Implement evaluation metrics (e.g., ROUGE, BLEU)

    Document example questions and expected outputs

## Chunking Strategy and Embedding Model Choice
# Chunking Strategy
Long consumer complaint narratives can often exceed the input limits or degrade embedding quality when processed as a single unit. To address this, we implemented a chunking strategy using LangChain’s RecursiveCharacterTextSplitter. This approach splits each narrative into smaller, semantically coherent chunks while preserving contextual overlap.

Chunk Size: We set the chunk size to 500 characters, which strikes a balance between retaining enough context and keeping chunks concise for embedding.

Chunk Overlap: An overlap of 50 characters was chosen to maintain continuity between chunks, reducing the risk of losing critical information at chunk boundaries.

Separators: The splitter uses a hierarchy of separators (\n\n, \n, ., !, ?, ,, space) to break text at natural linguistic boundaries, improving chunk quality.

This method ensures that each chunk represents a meaningful snippet of the original complaint, which enhances semantic embedding performance and downstream retrieval accuracy.

# Embedding Model Choice
For embedding the text chunks, we selected the sentence-transformers/all-MiniLM-L6-v2 model based on the following considerations:

Performance: This model is well-known for generating high-quality sentence embeddings that capture semantic similarity effectively, making it suitable for natural language search tasks.

Efficiency: It produces 384-dimensional vectors, which are compact enough to enable fast similarity computations and reduce storage overhead in the vector store.

Accessibility: Being part of the popular Sentence Transformers library, it integrates seamlessly with Python workflows and is optimized for GPU acceleration if available.

Together, the chunking strategy and embedding model prepare the complaint narratives for efficient and scalable semantic search in the Retrieval-Augmented Generation pipeline.


