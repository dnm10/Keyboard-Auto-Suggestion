"""
Automated Unit and Integration Tests for SmartNLP Keyboard Engine
"""

import unittest
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from nlp_engine import NLPEngine
from app import app
import textdistance


class TestNLPEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = NLPEngine()

    def test_corpus_loaded(self):
        stats = self.engine.get_stats()
        self.assertGreater(stats["total_tokens"], 100000)
        self.assertGreater(stats["vocab_size"], 10000)
        self.assertGreater(stats["unique_bigrams"], 50000)

    def test_h_prefix_suggestions(self):
        """Typing 'h' should suggest common conversational words like 'hi', 'hello', 'how'."""
        suggestions = [s["word"] for s in self.engine.suggest("h", top_n=5)]
        self.assertIn("hi", suggestions)
        self.assertIn("hello", suggestions)
        self.assertIn("how", suggestions)

    def test_next_word_after_hello(self):
        """Typing 'hello ' should predict 'my', 'how', 'there'."""
        suggestions = [s["word"] for s in self.engine.suggest("hello ", top_n=5)]
        self.assertIn("my", suggestions)
        self.assertIn("how", suggestions)

    def test_trigram_hello_my_predicts_name(self):
        """Typing 'hello my ' should strongly predict 'name'."""
        suggestions = self.engine.suggest("hello my ", top_n=5)
        top_word = suggestions[0]["word"]
        self.assertEqual(top_word, "name")

    def test_trigram_my_name_predicts_is(self):
        """Typing 'hello my name ' should predict 'is' with very high confidence (>90%)."""
        suggestions = self.engine.suggest("hello my name ", top_n=3)
        self.assertEqual(suggestions[0]["word"], "is")
        self.assertGreater(suggestions[0]["similarity"], 0.90)

    def test_autocorrect_helo(self):
        """Typing 'helo' should autocorrect to 'hello'."""
        suggestions = [s["word"] for s in self.engine.suggest("helo", top_n=3)]
        self.assertIn("hello", suggestions)

    def test_autocorrect_reciev(self):
        """Typing 'reciev' should autocorrect to 'receive'."""
        suggestions = [s["word"] for s in self.engine.get_spelling_corrections("reciev", top_n=5)]
        self.assertIn("receive", suggestions)

    def test_jaccard_math_accuracy(self):
        """Verify custom fast Jaccard matches textdistance library identically."""
        pairs = [
            ("hel", "help"),
            ("appl", "apple"),
            ("comput", "computer"),
            ("nat", "nature"),
            ("book", "book")
        ]
        for w1, w2 in pairs:
            expected = textdistance.Jaccard(qval=2).similarity(w1, w2)
            actual = self.engine.jaccard_similarity(w1, w2, q=2)
            self.assertAlmostEqual(actual, expected, places=5)


class TestFlaskEndpoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = app.test_client()

    def test_index_route(self):
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)

    def test_api_suggest_endpoint(self):
        res = self.client.get("/api/suggest?q=hello%20my%20name%20&mode=hybrid&top_n=3")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["suggestions"][0]["word"], "is")

    def test_api_stats_endpoint(self):
        res = self.client.get("/api/stats")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("total_tokens", data["stats"])


if __name__ == "__main__":
    unittest.main()
