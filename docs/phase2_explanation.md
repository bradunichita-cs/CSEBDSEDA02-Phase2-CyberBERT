## Phase 2 – Development & Reflection (Task 1: Cyber-BERT)

In this phase, the Cyber-BERT dataset consisting of short cybersecurity-related texts was loaded into Python using pandas and cleaned to reduce noise. Each text was converted to lowercase, URLs, digits, and punctuation were removed, repeated whitespace was normalized, and the artefact word “share” was removed because it appeared frequently in the dataset. Stop-words were handled automatically by the vectorization methods.

Two vectorization approaches were applied to transform the unstructured text into numerical features. First, Bag-of-Words (CountVectorizer) was used to create a document–term matrix based on token frequencies while filtering out very common terms. Second, TF–IDF (TfidfVectorizer) was applied to weight words based on their importance within documents relative to the entire corpus.

Two topic modelling algorithms were then trained. Latent Dirichlet Allocation (LDA) was applied to the count-based matrix to extract four general topics. Non-negative Matrix Factorisation (NMF) was trained on the TF–IDF matrix and produced more specific and interpretable topics related to vulnerabilities, malware, AI-based security tools, and cryptocurrency-related incidents. Overall, TF–IDF combined with NMF produced clearer topics for this small dataset, while LDA provided higher-level thematic groupings.

GitHub repository link: https://github.com/bradunichita-cs/CSEBDSEDA02-Phase2-CyberBERT
