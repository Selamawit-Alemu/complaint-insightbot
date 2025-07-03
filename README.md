# complaint-insightbot
## 📊 Exploratory Data Analysis (EDA) Summary
To understand the structure and quality of the CFPB complaint data, we conducted exploratory analysis on a 500,000-row sample. Among these, approximately 26,000 entries included a valid consumer complaint narrative. After filtering for the five targeted financial products — Credit Cards, Personal Loans, Buy Now, Pay Later (BNPL), Savings Accounts, and Money Transfers — we retained 1,720 high-quality records with detailed narratives suitable for downstream semantic retrieval and analysis.

The dataset revealed a highly imbalanced product distribution, with the majority of complaints centered around credit-related services. Narrative length varied significantly, ranging from very short (under 10 words) to over 5,000 words. However, most complaints fell within the 100–250 word range, which offers sufficient context for language model understanding without excessive verbosity. We also identified a large portion of entries with missing or boilerplate content, which were excluded to ensure clean, representative input for embedding.

As part of preprocessing, we applied standard text cleaning techniques, including lowercasing, special character removal, and elimination of repetitive template phrases. This ensures better embedding quality and consistency for the RAG pipeline. The cleaned and filtered dataset has been saved as data/filtered_complaints.csv, serving as the foundation for chunking, embedding, and semantic search in subsequent tasks.

