"""
Keyboard Auto-Suggestion & Autocorrect NLP Web Application
==========================================================
Flask Backend server providing:
1. Web interface (HTML5 + Jinja2)
2. Real-time REST API for live suggestions (/api/suggest)
3. Model diagnostics endpoint (/api/stats)
4. Backward-compatible Form Submission (/suggest)
"""

import os
from flask import Flask, render_template, request, jsonify
from nlp_engine import NLPEngine

# Resolve directory paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")
DATA_DIR = os.path.join(BASE_DIR, "data")
CORPUS_PATH = os.path.join(DATA_DIR, "autocorrect book.txt")
if not os.path.exists(CORPUS_PATH):
    CORPUS_PATH = os.path.join(BASE_DIR, "autocorrect book.txt")

app = Flask(
    __name__,
    template_folder=TEMPLATES_DIR,
    static_folder=STATIC_DIR
)

# Initialize NLP Engine with Corpus
print(f"[*] Initializing NLP Engine from corpus: {CORPUS_PATH}")
nlp = NLPEngine(corpus_path=CORPUS_PATH)
print(f"[+] Loaded vocabulary: {nlp.get_stats()}")


@app.route("/")
def index():
    """Renders the main interactive keyboard interface."""
    stats = nlp.get_stats()
    return render_template(
        "index.html",
        suggestions=None,
        keyword="",
        stats=stats
    )


@app.route("/suggest", methods=["POST"])
def suggest_form():
    """
    Backward-compatible form submission endpoint.
    Maintains full compatibility with original repository tests.
    """
    keyword = request.form.get("keyword", "").strip().lower()
    stats = nlp.get_stats()
    suggestions_list = []

    if keyword:
        # Fetch Top 10 Jaccard-based suggestions matching original logic
        raw_suggestions = nlp.get_jaccard_suggestions(keyword, top_n=10)
        suggestions_list = [
            {
                "Word": item["word"],
                "Similarity": item["similarity"],
                "Prob": item["probability"],
                "Freq": item["frequency"],
                "EditDist": item["edit_distance"]
            }
            for item in raw_suggestions
        ]

    return render_template(
        "index.html",
        suggestions=suggestions_list,
        keyword=keyword,
        stats=stats
    )


@app.route("/api/suggest", methods=["GET"])
def api_suggest():
    """
    Real-Time REST API Endpoint.
    Params:
      q: Input text or active prefix (e.g. 'hel', 'appl', 'the ')
      mode: 'hybrid' | 'jaccard' | 'levenshtein' | 'prefix' | 'next_word'
      top_n: Number of candidates to return (default 10)
    """
    query = request.args.get("q", "")
    mode = request.args.get("mode", "hybrid")
    try:
        top_n = min(max(int(request.args.get("top_n", 10)), 1), 30)
    except (ValueError, TypeError):
        top_n = 10

    candidates = nlp.suggest(text=query, mode=mode, top_n=top_n)

    return jsonify({
        "status": "success",
        "query": query,
        "mode": mode,
        "count": len(candidates),
        "suggestions": candidates
    })


@app.route("/api/stats", methods=["GET"])
def api_stats():
    """Returns engine diagnostics, vocab size, and corpus statistics."""
    return jsonify({
        "status": "success",
        "stats": nlp.get_stats()
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n[+] Server running at http://127.0.0.1:{port}")
    print("[*] Press Ctrl+C to stop.")
    app.run(host="0.0.0.0", port=port, debug=True)
