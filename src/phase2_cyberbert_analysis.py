"""
Phase 2 data analysis for the Cyber-BERT dataset.
"""

from pathlib import Path
import re
import string
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\\S+|www\\S+", "", text)
    text = re.sub(r"\\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\\bshare\\b", "", text)
    text = re.sub(r"\\s+", " ", text).strip()
    return text


def display_topics(model, feature_names, n_top_words=10):
    topics = []
    for topic in model.components_:
        top_indices = topic.argsort()[:-n_top_words - 1:-1]
        topics.append([feature_names[i] for i in top_indices])
    return topics


def main():
    file_path = Path(__file__).resolve().parents[1] / "data" / "cyberbert.csv"
    df = pd.read_csv(file_path)

    cleaned = df["text"].astype(str).apply(clean_text)

    count_vectorizer = CountVectorizer(stop_words="english", max_df=0.95)
    count_matrix = count_vectorizer.fit_transform(cleaned)

    tfidf_vectorizer = TfidfVectorizer(stop_words="english", max_df=0.95)
    tfidf_matrix = tfidf_vectorizer.fit_transform(cleaned)

    lda = LatentDirichletAllocation(n_components=4, random_state=0)
    lda.fit(count_matrix)

    nmf = NMF(n_components=4, random_state=1, init="nndsvd")
    nmf.fit(tfidf_matrix)

    print("LDA topics:")
    for i, topic in enumerate(
        display_topics(lda, count_vectorizer.get_feature_names_out()), 1
    ):
        print(f"Topic {i}: {', '.join(topic)}")

    print("\\nNMF topics:")
    for i, topic in enumerate(
        display_topics(nmf, tfi_
