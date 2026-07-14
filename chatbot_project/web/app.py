"""
app.py
Flask web server that exposes the same Python chatbot (data_loader,
preprocessor, classifier, chatbot_engine) over HTTP, so it can be used
from a browser-based chat UI (index.html) in Chrome.

Run:
    cd web
    python3 app.py
Then open http://127.0.0.1:5000 in Chrome.
"""

import os
import sys

# Make the src/ package importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from flask import Flask, request, jsonify, render_template

from data_loader import IntentDataLoader
from preprocessor import TextPreprocessor
from classifier import IntentClassifier
from chatbot_engine import ChatbotEngine

INTENTS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "intents.json")

app = Flask(__name__)

# One shared engine per server process. Each browser tab gets its own
# conversation history by sending a session_id; we keep a dict of engines.
_loader = IntentDataLoader(INTENTS_PATH)
_loader.load()
_texts, _labels = _loader.get_training_pairs()
_responses_map = _loader.get_responses_map()

_preprocessor = TextPreprocessor()
_classifier = IntentClassifier(_preprocessor, k_neighbors=3)
_train_results = _classifier.train(_texts, _labels)

_sessions = {}  # session_id -> ChatbotEngine


def get_engine(session_id):
    if session_id not in _sessions:
        _sessions[session_id] = ChatbotEngine(_classifier, _responses_map)
    return _sessions[session_id]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    payload = request.get_json(force=True) or {}
    message = (payload.get("message") or "").strip()
    session_id = payload.get("session_id", "default")

    if not message:
        return jsonify({"error": "Empty message"}), 400

    engine = get_engine(session_id)
    response, confidence = engine.get_response(message)
    last_intent = engine.history[-1]["intent"] if engine.history else None

    return jsonify({
        "response": response,
        "confidence": round(confidence, 2),
        "intent": last_intent,
    })


@app.route("/api/reset", methods=["POST"])
def reset():
    payload = request.get_json(force=True) or {}
    session_id = payload.get("session_id", "default")
    if session_id in _sessions:
        _sessions[session_id].reset()
    return jsonify({"status": "ok"})


@app.route("/api/model-info", methods=["GET"])
def model_info():
    return jsonify({
        "best_model": _classifier.best_model_name,
        "naive_bayes_accuracy": round(_train_results["naive_bayes"]["accuracy"], 2),
        "knn_accuracy": round(_train_results["knn"]["accuracy"], 2),
        "num_intents": len(_loader.get_tags()),
    })


if __name__ == "__main__":
    print("Model trained. Best model:", _classifier.best_model_name)
    print("Open http://127.0.0.1:5000 in Chrome.")
    
    app.run(
    host="0.0.0.0",
    port=int(os.environ.get("PORT", 5000)),
    debug=False
)
