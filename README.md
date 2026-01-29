# Phase 2 – Cyber-BERT Topic Modelling (CSEBDSEDA02)

This repository contains Phase 2 (Development & Reflection) for Task 1 of the IU course
**Project: Data Analysis (CSEBDSEDA02)**.

The project loads a small cybersecurity text dataset (Cyber-BERT CSV), cleans the text,
vectorizes it using two methods, and extracts topics using two topic-modelling algorithms.

## Project structure
- src/phase2_cyberbert_analysis.py – main analysis script
- data/cyberbert.csv – Cyber-BERT dataset
- requirements.txt – Python dependencies
- docs/phase2_explanation.md – Phase 2 explanation text

## Methods implemented
- Text cleaning (lowercasing, URL removal, punctuation and digit removal)
- Vectorization:
  - Bag-of-Words (CountVectorizer)
  - TF-IDF (TfidfVectorizer)
- Topic modelling:
  - LDA (Latent Dirichlet Allocation)
  - NMF (Non-negative Matrix Factorisation)

## Output
The script prints the top keywords for each extracted topic.
