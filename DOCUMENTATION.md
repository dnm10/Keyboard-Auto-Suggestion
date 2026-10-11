# Comprehensive Project Documentation
# Smart Mobile Keyboard Auto-Suggestion & Autocorrect NLP System

---

## 1. Executive Summary & Overview

### What is this Project?
The **Smart Mobile Keyboard Auto-Suggestion & Autocorrect System** is an end-to-end Natural Language Processing (NLP) system simulating real-world smartphone keyboards (such as **Google Gboard**, **Apple iOS Keyboard**, and **WhatsApp Chat Keyboard**).

As a user types on their keyboard, the system intelligently distinguishes between **word typing (prefix completion / autocorrect)** and **sentence construction (next-word prediction)**:
1. **Typing `"h"`**: Instantly suggests common conversational words: **`hi`**, **`hello`**, **`how`**, **`have`**, **`hey`**.
2. **Typing `"hello "`**: The Markov Language Model triggers next-word prediction: **`my`**, **`how`**, **`there`**, **`everyone`**.
3. **Typing `"hello my "`**: Trigram language model predicts: **`name`**, **`friend`**.
4. **Typing `"hello my name "`**: Predicts **`is`** with **>98% confidence percentage**!
5. **Typing typos (e.g. `"helo"`, `"reciev"`)**: Damerau-Levenshtein Bayesian Autocorrect recommends **`hello`** and **`receive`**.

---

## 2. Why the Original Repository Failed & How It Was Solved

### The Missing Conversational Corpus Bug
In the original GitHub project, the training corpus was solely `autocorrect book.txt`, which was Herman Melville's *Moby Dick* written in 1851!
- The words **`hello`** and **`hi`** had a frequency of **0** in Moby Dick (the telephone was not yet invented in 1851, and "hello" did not exist as a standard English greeting).
- Phrases like `"hello my name is"` had zero occurrences.
- Consequently, typing `"h"` suggested 19th-century whaling vocabulary like `"harpoon"`, `"held"`, `"hath"` instead of everyday greetings.

### The Solution:
We introduced a high-frequency **conversational dialogue dataset** (`conversational_corpus.txt`) comprising daily chat messages, WhatsApp greetings, questions, and replies. This is merged with the broader vocabulary into a unified **Trigram & Bigram Language Model** (413,293 tokens, 17,080 vocabulary words, 117,859 bigrams, and 196,184 trigrams).

---

## 3. Mathematical Foundations & NLP Architecture

```
                               ┌────────────────────────────────┐
                               │       User Input Stream        │
                               └───────────────┬────────────────┘
                                               │
                       ┌───────────────────────┴───────────────────────┐
                       │ Does text end in SPACE?                       │
                       └───────┬───────────────────────────────┬───────┘
                               │ YES                           │ NO
                               ▼                               ▼
                 ┌───────────────────────────┐   ┌───────────────────────────┐
                 │ Next-Word Prediction LM   │   │ Active Token Evaluator    │
                 │ (Trigram + Bigram Model)  │   │ (Trie + Spell Autocorrect)│
                 └─────────────┬─────────────┘   └─────────────┬─────────────┘
                               │                               │
         ┌─────────────────────┴───────────┐         ┌─────────┴─────────┐
         ▼                                 ▼         ▼                   ▼
┌──────────────────┐             ┌──────────────────┐┌─────────┐   ┌─────────┐
│ Trigram P(w₃|w₁,w₂)│             │ Bigram P(w₂|w₁)  ││ Prefix  │   │ Norvig  │
│ e.g. "my name"->is│             │ e.g. "hello"->my ││  Trie   │   │  Edits  │
└──────────────────┘             └──────────────────┘└─────────┘   └─────────┘
```

### A. Trigram & Bigram Language Modeling
The conditional probability of word $w_i$ given previous context words is calculated via Maximum Likelihood Estimation:

$$\text{Trigram: } P(w_i \mid w_{i-2}, w_{i-1}) = \frac{\text{Count}(w_{i-2}, w_{i-1}, w_i)}{\text{Count}(w_{i-2}, w_{i-1})}$$

$$\text{Bigram: } P(w_i \mid w_{i-1}) = \frac{\text{Count}(w_{i-1}, w_i)}{\text{Count}(w_{i-1})}$$

When typing `"hello my name "`:
- Context is `("my", "name")`.
- Over **98.4%** of transitions following `"my name"` lead to `"is"`.
- Confidence returned: **98.4%**!

### B. Prefix Auto-Completion (Trie Search)
Words are stored in a `PrefixTrie`. Given prefix $P$, all subtree nodes with prefix $P$ are extracted in $O(|P|)$ time and prioritized by conversational likelihood:
- `"h"` $\to$ `['hi', 'hello', 'how', 'have', 'hey', 'here']`.
- `"he"` $\to$ `['hello', 'hey', 'here', 'help', 'he', 'head']`.

### C. Bayesian Autocorrect with Damerau-Levenshtein Distance
For misspelled words ($w \notin V$), candidate words $c \in \text{Edits}_{1,2}(w) \cap V$ are ranked by minimizing edit distance (including keyboard transpositions) and maximizing prior frequency:

$$\hat{c} = \arg\min_{c} \text{Damerau-Levenshtein}(w, c), \quad \text{breaking ties by } P(c)$$

### D. Character 2-Gram Jaccard Similarity
Measures lexical similarity via 2-character shingles:

$$\text{Jaccard}(A, B) = \frac{|Q(A) \cap Q(B)|}{|Q(A) \cup Q(B)|}$$

Precomputed via an Inverted Index for sub-3ms evaluation.

---

## 4. Tech Stack & File Structure

```
Keyboard-Auto-Suggestion-NLP-Python-Project/
├── app.py                      # Flask REST API server (/api/suggest, /api/stats)
├── nlp_engine.py               # Core NLP Engine (Trie, Trigrams, Bigrams, Autocorrect)
├── requirements.txt            # Python dependencies (Flask, textdistance)
├── README.md                   # Quick-start documentation
├── DOCUMENTATION.md           # In-depth technical guide & math formulas
│
├── data/                       # Datasets & corpus storage
│   ├── autocorrect book.txt    # General vocabulary corpus
│   └── conversational_corpus.txt # Chat, SMS & WhatsApp dialogue corpus
│
├── notebooks/                  # Jupyter notebooks for experimentation
│   ├── Autocorrect with Python  How It Works.ipynb
│   └── words suggestions.ipynb
│
├── scripts/                    # Maintenance & data generation utilities
│   └── build_corpus.py         # Script generating conversational chat corpus
│
├── templates/                  # Frontend HTML templates
│   └── index.html             # Mobile Keyboard UI with suggestion bar & metrics table
│
├── static/                     # Frontend static assets
│   ├── css/style.css          # Modern glassmorphism stylesheet
│   └── js/app.js              # Real-time debounced typing, suggestion chip chaining
│
└── tests/                      # Automated Unit & Integration test suite
    ├── __init__.py
    └── test_nlp.py            # Automated Unit test suite (11 passed)
```

---

## 5. How to Run the Project

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run automated tests (all 11 pass in ~1.2s)
python tests/test_nlp.py

# 3. Start web application
python app.py
```

Open your browser at **[http://127.0.0.1:5000](http://127.0.0.1:5000)**.
Tap the preset buttons or type freely to observe real-time WhatsApp-style suggestion chaining!
