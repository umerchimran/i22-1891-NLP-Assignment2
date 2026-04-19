NLP Pipeline: Word Embeddings, POS Tagging & NER

This repository contains a complete Natural Language Processing pipeline implemented from scratch for CS-4063 NLP Assignment (FAST NUCES). The project includes word representation learning, sequence labeling, and evaluation using multiple neural and statistical methods.

📌 Project Overview

The project is divided into two main parts:

Part 1: Word Embeddings
TF-IDF based term-document representation
PPMI (Positive Pointwise Mutual Information) word embeddings
Skip-gram Word2Vec model (from scratch)
Nearest neighbour analysis and analogy testing
t-SNE visualization of word embeddings
Part 2: Sequence Labeling
POS Tagging using BiLSTM
Named Entity Recognition (NER) with BiLSTM + CRF
Frozen vs Fine-tuned embedding comparison
Ablation study on architecture components
📂 Project Structure
.
├── embeddings/
│   ├── tfidf_matrix.npy
│   ├── ppmi_matrix.npy
│   ├── embeddings_w2v.npy
│   ├── word2idx.json
│
├── results/
│   ├── pos_frozen_loss.png
│   ├── pos_finetune_loss.png
│   ├── ner_crf_loss.png
│   ├── ner_nocrf_loss.png
│   ├── pos_confusion_matrix.png
│
├── cleaned.txt
├── Metadata.json
├── part1_pipeline.py
├── part2_sequence_labeling.py
└── README.md
⚙️ Features
🔹 Word Embeddings
TF-IDF weighted representations
PPMI co-occurrence embeddings
Skip-gram Word2Vec (negative sampling)
Cosine similarity-based nearest neighbours
Word analogy evaluation
t-SNE visualization
🔹 Sequence Labeling
BiLSTM-based POS tagging
NER using BiLSTM + CRF
Support for frozen & fine-tuned embeddings
Dropout regularization
Viterbi decoding for CRF
📊 Results Summary
POS Tagging
Accuracy: ~99.9%
Best performance with fine-tuned embeddings
Strong results on NOUN and NUM tags
NER
Training loss converged to near zero
CRF vs Softmax: no major improvement observed
Poor test F1 due to dataset/annotation limitations
Word Embeddings
Word2Vec outperformed TF-IDF and PPMI
PPMI struggled due to sparsity
TF-IDF useful for keyword extraction only
🧠 Key Insights
Word2Vec captures strong semantic relationships
Bidirectional LSTM improves sequence understanding
Pretrained embeddings significantly improve convergence
CRF helps structured prediction but depends on data quality
🚀 How to Run
1. Install dependencies
pip install numpy torch scikit-learn matplotlib
2. Run Part 1 (Embeddings)
python part1_pipeline.py
3. Run Part 2 (Sequence Labeling)
python part2_sequence_labeling.py
📌 Technologies Used
Python
PyTorch
NumPy
Scikit-learn
Matplotlib
