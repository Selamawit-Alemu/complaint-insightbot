# complaint-insightbot
## 📊 Exploratory Data Analysis (EDA) Summary
To understand the structure and quality of the CFPB complaint data, we conducted exploratory analysis on a 500,000-row sample. Among these, approximately 26,000 entries included a valid consumer complaint narrative. After filtering for the five targeted financial products — Credit Cards, Personal Loans, Buy Now, Pay Later (BNPL), Savings Accounts, and Money Transfers — we retained 1,720 high-quality records with detailed narratives suitable for downstream semantic retrieval and analysis.

The dataset revealed a highly imbalanced product distribution, with the majority of complaints centered around credit-related services. Narrative length varied significantly, ranging from very short (under 10 words) to over 5,000 words. However, most complaints fell within the 100–250 word range, which offers sufficient context for language model understanding without excessive verbosity. We also identified a large portion of entries with missing or boilerplate content, which were excluded to ensure clean, representative input for embedding.

As part of preprocessing, we applied standard text cleaning techniques, including lowercasing, special character removal, and elimination of repetitive template phrases. This ensures better embedding quality and consistency for the RAG pipeline. The cleaned and filtered dataset has been saved as data/filtered_complaints.csv, serving as the foundation for chunking, embedding, and semantic search in subsequent tasks.

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


