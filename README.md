Student ID: 22I-1891
Name: Umer Imran
Course: CS-4063 – Natural Language Processing

This repository contains my complete submission for Assignment 2 of the NLP course. The objective of this assignment was to design and implement a full neural NLP pipeline from scratch in PyTorch, covering:

Word embeddings learning
Sequence labeling (POS + NER)
Transformer-based topic classification

The entire system was built on a BBC Urdu corpus, following strict constraints that required no use of pre-trained models, Gensim, HuggingFace, or built-in PyTorch Transformer modules. All components were implemented manually using raw PyTorch tensor operations.

Repository Contents
22I-1891_Assignment2_DS-C.ipynb
Main Jupyter Notebook containing the complete implementation of all three parts of the assignment.
22I-1891_Assignment2_Report.pdf
Detailed report covering methodology, experimental setup, results, ablation studies, and analysis.
cleaned.txt & raw.txt
BBC Urdu corpus used for training and evaluation.
Metadata.json
Topic-category mapping used for transformer classification.
/embeddings

Contains all generated representation files:

TF-IDF matrices
PPMI co-occurrence embeddings
Skip-gram Word2Vec embeddings (.npy)
t-SNE visualization outputs (.png)
/models

Contains trained model checkpoints:

BiLSTM POS tagger (frozen + fine-tuned versions)
BiLSTM + CRF NER model
Final Transformer encoder classification model
How to Run
1. Setup Environment

Install required dependencies:

pip install torch numpy matplotlib scikit-learn scipy jupyterlab
2. Launch Notebook

Run the following command in the project root:

jupyter lab 22I-1891_Assignment2_DS-C.ipynb
3. Execution Flow

The notebook is structured into three main sections:

Part 1: Word Embeddings
TF-IDF and PPMI representation
Skip-gram Word2Vec model
Nearest neighbour & analogy evaluation
t-SNE visualization
Part 2: Sequence Labeling
Rule-based POS & NER preprocessing
BiLSTM sequence labeling
CRF-based decoding for NER
Frozen vs fine-tuned embeddings comparison
Part 3: Transformer Encoder
Scaled dot-product attention
Multi-head attention (from scratch)
Sinusoidal positional encoding
4-layer Transformer encoder
5-class topic classification
Implementation Overview
Part 1: Word Embeddings

Implemented TF-IDF, PPMI, and Skip-gram Word2Vec from scratch to learn word representations and evaluate semantic similarity using cosine-based methods.

Part 2: Sequence Labeling

Built a BiLSTM-based POS and NER system with CRF decoding. Evaluated frozen vs fine-tuned embeddings and performed ablation studies.

Part 3: Transformer Encoder

Implemented a full Transformer encoder manually, including multi-head attention, positional encoding, and classification head for topic classification.

Report & Results

For detailed results, training curves, and analysis, refer to:

📄 22I-1891_Assignment2_Report.pdf

Notes
No pre-trained models or external NLP frameworks were used
All models were implemented from scratch using PyTorch
Dataset: BBC Urdu Corpus
Fully compliant with assignment restrictions
Author

Name: Umer Imran
Roll Number: 22I-1891
Course: CS-4063 – Natural Language Processing
University: FAST NUCES
