"""Flask demo application for BERT Question Answering."""

import os
import sys

# Add parent directory to path so we can import src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, render_template, request, jsonify

from src.model import BERTQA

app = Flask(__name__)

# Lazy-loaded model singleton
_model = None


def get_model():
    """Get or initialize the BERT QA model."""
    global _model
    if _model is None:
        _model = BERTQA()
    return _model


@app.route("/", methods=["GET", "POST"])
def index():
    """Render main page and handle QA form submission."""
    context = {
        "answer": None,
        "confidence": None,
        "highlighted_passage": None,
        "error": None,
        "request": request,
        "model_info": None,
    }

    if request.method == "POST":
        passage = request.form.get("passage", "").strip()
        question = request.form.get("question", "").strip()
        chunk_size = request.form.get("chunk_size", "384")
        stride = request.form.get("stride", "128")

        if not passage:
            context["error"] = "Please enter a passage."
            return render_template("index.html", **context)
        if not question:
            context["error"] = "Please enter a question."
            return render_template("index.html", **context)

        try:
            chunk_size = int(chunk_size)
            stride = int(stride)
        except ValueError:
            context["error"] = "Invalid chunk size or stride."
            return render_template("index.html", **context)

        model = get_model()
        result = model.answer(
            question=question,
            context=passage,
            chunk_size=chunk_size,
            stride=stride,
        )

        answer = result.get("answer", "")
        score = result.get("score", 0.0)
        start_idx = result.get("start", 0)
        end_idx = result.get("end", 0)

        info = model.info()
        context["model_info"] = {
            "name": info["model_name"].split("/")[-1],
            "params": f"{info['num_parameters']:,}",
            "is_micro": info["is_micro"],
        }

        if answer:
            context["answer"] = answer
            context["confidence"] = f"{score:.4f}"

            # Highlight answer in passage
            if 0 <= start_idx < len(passage) and 0 <= end_idx <= len(passage):
                before = passage[:start_idx]
                highlight = passage[start_idx:end_idx]
                after = passage[end_idx:]
                context["highlighted_passage"] = (before, highlight, after)
            else:
                context["highlighted_passage"] = (passage, "", "")
        else:
            context["error"] = "No answer found. Try rephrasing your question."

    return render_template("index.html", **context)


@app.route("/api/answer", methods=["POST"])
def api_answer():
    """REST API endpoint for question answering."""
    data = request.get_json(force=True)
    passage = data.get("passage", "").strip()
    question = data.get("question", "").strip()
    chunk_size = data.get("chunk_size", 384)
    stride = data.get("stride", 128)

    if not passage or not question:
        return jsonify({"error": "Both passage and question are required."}), 400

    model = get_model()
    result = model.answer(
        question=question,
        context=passage,
        chunk_size=chunk_size,
        stride=stride,
    )

    info = model.info()
    return jsonify({
        "answer": result.get("answer", ""),
        "confidence": result.get("score", 0.0),
        "start": result.get("start", 0),
        "end": result.get("end", 0),
        "model_params": info["num_parameters"],
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
