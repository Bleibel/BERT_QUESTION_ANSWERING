"""Flask demo application for BERT Question Answering."""

import os
import sys

# Add parent directory to path so we can import src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, render_template, request, jsonify

from src.model import BERTQA

app = Flask(__name__)

# Caching mechanism for multiple loaded checkpoints
_models = {}


def list_checkpoints():
    """Dynamically scan the checkpoints directory and return available models."""
    checkpoints = []

    base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints")
    if os.path.exists(base_dir):
        for root, dirs, files in os.walk(base_dir):
            # Check if directory contains model files or config
            if "config.json" in files:
                rel_path = os.path.relpath(root, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                rel_path = rel_path.replace("\\", "/")
                
                # Format name nicely
                name = os.path.basename(root).replace("-", " ").title()
                if "ppo" in rel_path.lower():
                    name += " (PPO)"
                elif "rl" in rel_path.lower():
                    name += " (REINFORCE)"
                elif "distill" in rel_path.lower():
                    name += " (Distilled)"
                
                is_micro = "micro" in name.lower() or "micro" in rel_path.lower()
                checkpoints.append({
                    "name": name,
                    "path": rel_path,
                    "is_micro": is_micro
                })

    # Always ensure the default micro-bert checkpoint is listed
    has_default = any("micro-bert-qa" in cp["path"] for cp in checkpoints)
    if not has_default:
        checkpoints.append({
            "name": "Micro-BERT QA (Default)",
            "path": "checkpoints/micro-bert-qa",
            "is_micro": True
        })

    return checkpoints


def get_model(model_name=None):
    """Retrieve or initialize a cached model based on checkpoint path."""
    global _models
    
    # Resolve default path if None
    if model_name is None:
        model_name = "checkpoints/micro-bert-qa"

    if model_name not in _models:
        print(f"Loading BERTQA model checkpoint from: {model_name}...")
        _models[model_name] = BERTQA(model_name=model_name)
    return _models[model_name]


@app.route("/", methods=["GET", "POST"])
def index():
    """Render main page and handle QA form submission."""
    DEFAULT_PASSAGE = (
        "Berlin is the capital and largest city of Germany by both area and population. "
        "Its 3.7 million inhabitants make it the European Union's most populous city, "
        "according to population within city limits. The city is also one of Germany's 16 "
        "federal states. It is surrounded by the state of Brandenburg, and contiguous with "
        "Potsdam, Brandenburg's capital. Berlin's urban area has a population of around 4.5 "
        "million and is the second most populous urban area in Germany after the Ruhr."
    )

    available_checkpoints = list_checkpoints()
    selected_checkpoint = request.form.get("model_name", available_checkpoints[0]["path"])

    context = {
        "answer": None,
        "confidence": None,
        "error": None,
        "request": request,
        "model_info": None,
        "available_checkpoints": available_checkpoints,
        "selected_checkpoint": selected_checkpoint,
    }

    if request.method == "POST":
        passage = DEFAULT_PASSAGE
        question = request.form.get("question", "").strip()
        chunk_size = request.form.get("chunk_size", "384")
        stride = request.form.get("stride", "128")

        if not question:
            context["error"] = "Please enter a question."
            return render_template("index.html", **context)

        try:
            chunk_size = int(chunk_size)
            stride = int(stride)
        except ValueError:
            context["error"] = "Invalid chunk size or stride."
            return render_template("index.html", **context)

        try:
            model = get_model(selected_checkpoint)
            result = model.answer(
                question=question,
                context=passage,
                chunk_size=chunk_size,
                stride=stride,
            )

            answer = result.get("answer", "")
            score = result.get("score", 0.0)

            info = model.info()
            context["model_info"] = {
                "name": info["model_name"].split("/")[-1],
                "params": f"{info['num_parameters']:,}",
                "is_micro": info["is_micro"],
            }

            if answer:
                context["answer"] = answer
                context["confidence"] = f"{score:.4f}"
            else:
                context["error"] = "No answer found. Try rephrasing your question."
        except Exception as e:
            context["error"] = f"Model error: {str(e)}"

    return render_template("index.html", **context)


@app.route("/api/answer", methods=["POST"])
def api_answer():
    """REST API endpoint for question answering."""
    data = request.get_json(force=True)
    passage = data.get("passage", "").strip()
    question = data.get("question", "").strip()
    chunk_size = data.get("chunk_size", 384)
    stride = data.get("stride", 128)
    model_name = data.get("model_name", None)

    if not passage or not question:
        return jsonify({"error": "Both passage and question are required."}), 400

    try:
        model = get_model(model_name)
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
    except Exception as e:
        return jsonify({"error": f"Model error: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
