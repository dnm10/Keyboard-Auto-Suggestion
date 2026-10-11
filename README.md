# ⌨️ Smart Mobile Keyboard Auto-Suggestion & Autocorrect NLP System

A fast, end-to-end Natural Language Processing (NLP) web system simulating modern smartphone keyboards (Gboard / iOS / WhatsApp) with real-time auto-completion, Bayesian spelling autocorrect, character 2-gram Jaccard similarity, and Trigram/Bigram next-word prediction.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![Flask](https://img.shields.io/badge/Flask-3.1-green.svg)
![Tests](https://img.shields.io/badge/Tests-11%20Passed-brightgreen.svg)
![Latency](https://img.shields.io/badge/Latency-%3C4ms-purple.svg)

---

## 📁 Project Architecture & Folder Layout

```
Keyboard-Auto-Suggestion-NLP-Python-Project/
├── app.py                      # Flask Backend server with REST API (/api/suggest, /api/stats)
├── nlp_engine.py               # Core NLP Engine (Trie, Trigram & Bigram LM, Autocorrect)
├── requirements.txt            # Lightweight dependencies (Flask, textdistance)
├── README.md                   # Project overview & quick start
├── DOCUMENTATION.md           # In-depth technical documentation & math formulas
│
├── data/                       # Dataset & Corpus storage
│   ├── autocorrect book.txt    # General English vocabulary corpus
│   └── conversational_corpus.txt # Chat, SMS & WhatsApp dialogue corpus
│
├── notebooks/                  # Jupyter notebooks for experimentation & exploration
│   ├── Autocorrect with Python  How It Works.ipynb
│   └── words suggestions.ipynb
│
├── scripts/                    # Maintenance & data generation utilities
│   └── build_corpus.py         # Script generating conversational chat corpus
│
├── templates/                  # Frontend HTML templates
│   └── index.html             # Glassmorphic web UI with virtual keyboard & stats
│
├── static/                     # Frontend static assets
│   ├── css/
│   │   └── style.css          # Modern dark glassmorphic stylesheet
│   └── js/
│       └── app.js             # Real-time debounced typing, suggestion chip chaining
│
└── tests/                      # Automated Unit & Integration test suite
    ├── __init__.py
    └── test_nlp.py            # 11 comprehensive automated test cases
```

---

## 🌟 Key Features

- **⚡ Real-Time Auto-Completion**: Prefix Trie with $O(k)$ lookups (e.g. typing `'h'` suggests `'hi'`, `'hello'`, `'how'`, `'have'`, `'hey'`).
- **🔮 WhatsApp / Mobile Next-Word Prediction**: Trigram Markov Language Model ($P(w_3 \mid w_1, w_2)$) predicting continuations (e.g. `'hello '` $\to$ `'my'`, `'hello my '` $\to$ `'name'`, `'hello my name '` $\to$ `'is'` with **98.4% confidence**).
- **✨ Bayesian Spell Autocorrect**: Peter Norvig candidate generator using Damerau-Levenshtein distance (handles adjacent letter transpositions).
- **📐 Character 2-Gram Jaccard Metric**: Fast character n-gram overlap accelerated with an Inverted Index.
- **📱 One-Tap Sentence Chaining**: Tap suggestion chips to automatically insert words and trigger the next word predictions.
- **🔌 High-Speed REST API**: `/api/suggest` with $<4$ ms inference latency.

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Automated Tests
```bash
python tests/test_nlp.py
```

### 3. Launch the Application
```bash
python app.py
```
Open **[http://127.0.0.1:5000](http://127.0.0.1:5000)** in your web browser.

---

## 📚 Complete Documentation

Read **[DOCUMENTATION.md](DOCUMENTATION.md)** for mathematical equations, algorithm breakdowns, and API reference.
