"""
SmartNLP Keyboard Engine
========================
High-performance NLP Engine simulating smartphone keyboards (Gboard / iOS / WhatsApp).
Features:
- Conversational Trigram & Bigram Language Model for smart next-word prediction
- Instant Prefix Trie for auto-completion (e.g., 'h' -> 'hi', 'hello', 'how')
- Damerau-Levenshtein Edit Distance & Norvig Bayesian Autocorrect
- Character 2-gram Jaccard Similarity with inverted index acceleration
"""

import os
import re
from collections import Counter, defaultdict
import textdistance


class TrieNode:
    """Trie node for fast prefix auto-completion."""
    __slots__ = ('children', 'is_word', 'word', 'frequency')

    def __init__(self):
        self.children = {}
        self.is_word = False
        self.word = None
        self.frequency = 0


class PrefixTrie:
    """Prefix Trie for instant O(k) lookups."""
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str, frequency: int):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_word = True
        node.word = word
        node.frequency = frequency

    def search_prefix(self, prefix: str, max_results: int = 12):
        """Returns vocabulary words starting with prefix, ranked by frequency."""
        node = self.root
        prefix = prefix.lower()
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]

        results = []
        def _dfs(curr):
            if curr.is_word:
                results.append((curr.word, curr.frequency))
            for child in curr.children.values():
                _dfs(child)

        _dfs(node)
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:max_results]


class NLPEngine:
    """Core NLP Engine powering Smartphone Keyboard Auto-Suggestion & Autocorrect."""

    def __init__(self, corpus_path=None, corpus_files=None):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(base_dir, "data")
        
        def resolve_file(filename):
            """Locates file in data/ directory or base directory."""
            p_data = os.path.join(data_dir, filename)
            if os.path.exists(p_data):
                return p_data
            p_base = os.path.join(base_dir, filename)
            if os.path.exists(p_base):
                return p_base
            return p_data

        self.corpus_files = []
        if corpus_files:
            self.corpus_files.extend(corpus_files)
        elif corpus_path:
            if not os.path.exists(corpus_path):
                # Check inside data/
                alt_path = os.path.join(data_dir, os.path.basename(corpus_path))
                if os.path.exists(alt_path):
                    corpus_path = alt_path
            self.corpus_files.append(corpus_path)
            conv_path = resolve_file("conversational_corpus.txt")
            if os.path.exists(conv_path) and conv_path != corpus_path:
                self.corpus_files.insert(0, conv_path)
        else:
            conv_path = resolve_file("conversational_corpus.txt")
            book_path = resolve_file("autocorrect book.txt")
            if os.path.exists(conv_path):
                self.corpus_files.append(conv_path)
            if os.path.exists(book_path):
                self.corpus_files.append(book_path)

        self.words = []
        self.vocab = Counter()
        self.total_tokens = 0
        self.probs = {}
        self.vocab_set = set()
        self.trie = PrefixTrie()
        
        # Language Models
        self.bigram_model = defaultdict(Counter)
        self.trigram_model = defaultdict(Counter)
        
        # Inverted index for fast Jaccard scoring
        self.vocab_qgrams = {}
        self.qgram_inverted_index = defaultdict(list)

        # Conversational Priority Boost for smartphone keyboards
        self.chat_priority = {
            "hi": 1000000, "hello": 950000, "how": 900000, "have": 850000,
            "hey": 800000, "here": 750000, "help": 700000, "name": 900000,
            "is": 950000, "my": 920000, "you": 940000, "are": 930000,
            "what": 910000, "where": 860000, "when": 840000, "why": 820000,
            "who": 800000, "please": 880000, "thank": 870000, "thanks": 875000,
            "good": 890000, "morning": 810000, "night": 805000, "today": 830000,
            "tomorrow": 800000, "doing": 850000, "going": 840000, "call": 820000,
            "send": 815000, "fine": 800000, "great": 825000, "nice": 820000,
            "meet": 810000, "friend": 830000, "time": 820000
        }
        
        self.load_corpus()

    def load_corpus(self):
        """Builds unigram, bigram, trigram models and indexes from corpus files."""
        tokens = []
        for file_path in self.corpus_files:
            if os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    tokens.extend(re.findall(r"\b[a-z]+\b", f.read().lower()))

        if not tokens:
            raise FileNotFoundError("Corpus file(s) not found or empty.")

        self.words = tokens
        self.total_tokens = len(tokens)
        self.vocab = Counter(tokens)
        self.vocab_set = set(self.vocab.keys())

        # Unigrams & Trie
        for word, count in self.vocab.items():
            self.probs[word] = count / self.total_tokens
            self.trie.insert(word, count)

        # Bigrams: (w1) -> w2
        for w1, w2 in zip(self.words[:-1], self.words[1:]):
            self.bigram_model[w1][w2] += 1

        # Trigrams: (w1, w2) -> w3
        for w1, w2, w3 in zip(self.words[:-2], self.words[1:-1], self.words[2:]):
            self.trigram_model[(w1, w2)][w3] += 1

        # Precompute character 2-grams (q=2)
        for word in self.vocab_set:
            qg = self._extract_qgrams(word, q=2)
            self.vocab_qgrams[word] = qg
            for gram in qg:
                self.qgram_inverted_index[gram].append(word)

    @staticmethod
    def _extract_qgrams(word: str, q: int = 2):
        """Extracts character 2-grams from a word."""
        if len(word) < q:
            return {word}
        return {word[i:i + q] for i in range(len(word) - q + 1)}

    def jaccard_similarity(self, word1: str, word2: str, q: int = 2) -> float:
        """Computes exact 2-gram Jaccard Similarity: |A ∩ B| / |A ∪ B|."""
        q1 = self._extract_qgrams(word1, q)
        q2 = self.vocab_qgrams.get(word2) or self._extract_qgrams(word2, q)
        inter = len(q1 & q2)
        union = len(q1 | q2)
        return inter / union if union > 0 else 0.0

    def edit_distance(self, s1: str, s2: str) -> int:
        """Damerau-Levenshtein distance accounting for keyboard transpositions."""
        return int(textdistance.damerau_levenshtein.distance(s1, s2))

    def norvig_edits1(self, word: str):
        """Generates all strings 1 edit away from word."""
        letters = "abcdefghijklmnopqrstuvwxyz"
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes = [L + R[1:] for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
        replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
        inserts = [L + c + R for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)

    def norvig_known_edits2(self, word: str):
        """Optimized: Generates only edit-2 candidates that exist in the vocabulary."""
        e1 = self.norvig_edits1(word)
        return {e2 for e1_w in e1 for e2 in self.norvig_edits1(e1_w) if e2 in self.vocab_set}

    def get_prefix_completions(self, prefix: str, top_n: int = 6):
        """
        Instant prefix auto-completions for what the user is typing.
        e.g., 'h' -> 'hi', 'hello', 'how', 'have', 'hey', 'here', 'help'
        """
        prefix = prefix.strip().lower()
        if not prefix:
            return []

        raw_matches = self.trie.search_prefix(prefix, max_results=top_n * 3)
        if not raw_matches:
            return []

        # Filter out 1-letter abbreviations (except valid standalone words 'a' and 'i')
        valid_matches = [
            (w, freq) for w, freq in raw_matches
            if (len(w) > 1 or w in {"a", "i"} or len(prefix) == len(w))
        ]

        def rank_key(item):
            w, freq = item
            return self.chat_priority.get(w, 0) + freq

        valid_matches.sort(key=rank_key, reverse=True)
        results = []
        for word, freq in valid_matches[:top_n]:
            ratio = len(prefix) / len(word)
            results.append({
                "word": word,
                "similarity": 1.0 if word == prefix else round(ratio, 2),
                "probability": self.probs.get(word, 0.0),
                "frequency": freq,
                "edit_distance": 0 if word == prefix else len(word) - len(prefix),
                "type": "completion"
            })
        return results

    def get_spelling_corrections(self, word: str, top_n: int = 5):
        """Autocorrects typos using Peter Norvig's Bayesian approach & Damerau-Levenshtein."""
        word = word.strip().lower()
        if not word:
            return []

        candidates = set()
        if word in self.vocab_set:
            candidates.add(word)

        e1_valid = self.norvig_edits1(word) & self.vocab_set
        candidates.update(e1_valid)

        if len(candidates) < top_n:
            e2_valid = self.norvig_known_edits2(word)
            candidates.update(e2_valid)

        if not candidates:
            return self.get_jaccard_suggestions(word, top_n=top_n)

        scored = []
        for cand in candidates:
            dist = self.edit_distance(word, cand)
            sim = self.jaccard_similarity(word, cand)
            score = self.chat_priority.get(cand, 0) + self.vocab.get(cand, 0)
            scored.append({
                "word": cand,
                "similarity": round(sim, 4),
                "probability": self.probs.get(cand, 0.0),
                "frequency": self.vocab.get(cand, 0),
                "edit_distance": dist,
                "score": score,
                "type": "autocorrect"
            })

        # Rank: least edit distance first, highest score second
        scored.sort(key=lambda x: (x["edit_distance"], -x["score"]))
        return scored[:top_n]

    def get_jaccard_suggestions(self, keyword: str, top_n: int = 5):
        """Fast Jaccard 2-gram similarity matching."""
        keyword = keyword.strip().lower()
        if not keyword:
            return []

        q_query = self._extract_qgrams(keyword, q=2)
        candidates = set()
        for g in q_query:
            candidates.update(self.qgram_inverted_index.get(g, []))

        if not candidates:
            candidates = {w for w in self.vocab_set if w[0] == keyword[0]} or set(list(self.vocab_set)[:500])

        results = []
        for word in candidates:
            sim = self.jaccard_similarity(keyword, word, q=2)
            if sim > 0:
                results.append({
                    "word": word,
                    "similarity": round(float(sim), 4),
                    "probability": self.probs[word],
                    "frequency": self.vocab[word],
                    "edit_distance": self.edit_distance(keyword, word),
                    "type": "jaccard_similarity"
                })

        results.sort(key=lambda x: (x["similarity"], x["frequency"]), reverse=True)
        return results[:top_n]

    def predict_next_word(self, context_tokens, top_n: int = 5):
        """
        Smartphone Trigram Language Model with Bigram fallback.
        Examples:
          ['hello'] -> 'my', 'how', 'there'
          ['hello', 'my'] -> 'name', 'friend'
          ['my', 'name'] -> 'is' (>95% confidence)
          ['name', 'is'] -> 'john', 'alex', 'david'
        """
        if not context_tokens:
            top_words = [("hi", 0.40), ("hello", 0.35), ("how", 0.25), ("i", 0.20), ("what", 0.15)]
            return [
                {"word": w, "similarity": prob, "probability": prob, "frequency": self.vocab.get(w, 100), "edit_distance": 0, "type": "starter"}
                for w, prob in top_words[:top_n]
            ]

        candidates = Counter()

        # 1. Trigram match: check last two words (w1, w2)
        has_trigrams = False
        if len(context_tokens) >= 2:
            w1, w2 = context_tokens[-2], context_tokens[-1]
            if (w1, w2) in self.trigram_model:
                for w3, cnt in self.trigram_model[(w1, w2)].items():
                    candidates[w3] += cnt * 6
                has_trigrams = True

        # 2. Bigram fallback: check last word (only if trigram has few results)
        if not has_trigrams or len(candidates) < top_n:
            last_word = context_tokens[-1]
            if last_word in self.bigram_model:
                for w2, cnt in self.bigram_model[last_word].items():
                    candidates[w2] += cnt

        if not candidates:
            for w, cnt in self.vocab.most_common(top_n):
                candidates[w] += 1

        total = sum(candidates.values())
        top_cands = candidates.most_common(top_n)

        return [
            {
                "word": w,
                "similarity": round(cnt / total, 3),
                "probability": self.probs.get(w, cnt / total),
                "frequency": self.vocab.get(w, cnt),
                "edit_distance": 0,
                "type": "next_word"
            }
            for w, cnt in top_cands
        ]

    def suggest(self, text: str, mode: str = "hybrid", top_n: int = 5):
        """
        Unified smartphone keyboard suggestion handler.
        Distinguishes active word typing (completions + autocorrect)
        from word completion (next-word predictions).
        """
        if not text:
            return self.predict_next_word([], top_n=top_n)

        tokens = re.findall(r"\b[a-z]+\b", text.lower())

        # If text ends with a space: user finished a word -> Next-Word Prediction
        if text.endswith(" "):
            if tokens:
                return self.predict_next_word(tokens, top_n=top_n)
            return self.suggest("", mode=mode, top_n=top_n)

        current_token = tokens[-1] if tokens else text.strip().lower()

        if mode == "jaccard":
            return self.get_jaccard_suggestions(current_token, top_n=top_n)
        elif mode == "levenshtein" or mode == "autocorrect":
            return self.get_spelling_corrections(current_token, top_n=top_n)
        elif mode == "prefix":
            return self.get_prefix_completions(current_token, top_n=top_n)
        elif mode == "next_word":
            context = tokens[:-1] if len(tokens) > 1 else [current_token]
            return self.predict_next_word(context, top_n=top_n)

        # Hybrid Mode (Smartphone / WhatsApp style):
        # 1. Prefix completions (e.g. 'h' -> 'hi', 'hello', 'how', 'have', 'hey')
        completions = self.get_prefix_completions(current_token, top_n=top_n)

        # 2. Autocorrect typos (only if length >= 3 and not an exact match)
        corrections = []
        if len(current_token) >= 3 and current_token not in self.vocab_set:
            corrections = self.get_spelling_corrections(current_token, top_n=3)

        seen = set()
        merged = []

        # If it's a typo, autocorrect comes first
        for item in corrections:
            w = item["word"]
            if w not in seen:
                seen.add(w)
                merged.append(item)

        # Prefix completions come next
        for item in completions:
            w = item["word"]
            if w not in seen:
                seen.add(w)
                merged.append(item)

        # If user typed an exact completed word (e.g. 'hello'), also show top next words
        if current_token in self.vocab_set:
            next_words = self.predict_next_word(tokens, top_n=3)
            for nw in next_words:
                if nw["word"] not in seen:
                    seen.add(nw["word"])
                    merged.append(nw)

        return merged[:top_n]

    def get_stats(self):
        """Returns vocabulary diagnostics."""
        return {
            "total_tokens": self.total_tokens,
            "vocab_size": len(self.vocab),
            "unique_bigrams": sum(len(v) for v in self.bigram_model.values()),
            "unique_trigrams": sum(len(v) for v in self.trigram_model.values())
        }
