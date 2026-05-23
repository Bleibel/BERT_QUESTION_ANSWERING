"""Dataset utilities for SQuAD-style reading comprehension data."""

import json
import random
from typing import Dict, List


def load_squad_data(file_path: str) -> List[dict]:
    """Load SQuAD v1.1 or v2.0 JSON data.

    Returns a flat list of examples, each with keys:
        id, title, context, question, answer_text, answer_start
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    examples = []
    for article in data.get("data", []):
        title = article.get("title", "")
        for paragraph in article.get("paragraphs", []):
            context = paragraph.get("context", "")
            for qa in paragraph.get("qas", []):
                qid = qa["id"]
                question = qa["question"]
                is_impossible = qa.get("is_impossible", False)

                if is_impossible:
                    # Skip impossible questions for standard QA evaluation
                    continue

                answers = qa.get("answers", [])
                if not answers:
                    continue

                # Use first answer
                answer = answers[0]
                examples.append({
                    "id": qid,
                    "title": title,
                    "context": context,
                    "question": question,
                    "answer_text": answer["text"],
                    "answer_start": answer["answer_start"],
                })

    return examples


def get_sample_dataset() -> List[dict]:
    """Return a small built-in sample dataset for quick testing."""
    return [
        {
            "id": "sample_1",
            "title": "Berlin",
            "context": (
                "Berlin is the capital and largest city of Germany by both area and population. "
                "Its 3.7 million inhabitants make it the European Union's most populous city, "
                "according to population within city limits. The city is also one of Germany's "
                "16 federal states. It is surrounded by the state of Brandenburg, and contiguous "
                "with Potsdam, Brandenburg's capital. Berlin's urban area has a population of "
                "around 4.5 million and is the second most populous urban area in Germany after "
                "the Ruhr. The Berlin-Brandenburg capital region has over 6 million inhabitants "
                "and is Germany's third-largest metropolitan region."
            ),
            "question": "What is the capital of Germany?",
            "answer_text": "Berlin",
            "answer_start": 0,
        },
        {
            "id": "sample_2",
            "title": "Berlin",
            "context": (
                "Berlin is the capital and largest city of Germany by both area and population. "
                "Its 3.7 million inhabitants make it the European Union's most populous city, "
                "according to population within city limits. The city is also one of Germany's "
                "16 federal states. It is surrounded by the state of Brandenburg, and contiguous "
                "with Potsdam, Brandenburg's capital. Berlin's urban area has a population of "
                "around 4.5 million and is the second most populous urban area in Germany after "
                "the Ruhr. The Berlin-Brandenburg capital region has over 6 million inhabitants "
                "and is Germany's third-largest metropolitan region."
            ),
            "question": "How many inhabitants does Berlin have?",
            "answer_text": "3.7 million",
            "answer_start": 83,
        },
        {
            "id": "sample_3",
            "title": "Photosynthesis",
            "context": (
                "Photosynthesis is a process used by plants and other organisms to convert light "
                "energy into chemical energy that, through cellular respiration, can later be "
                "released to fuel the organism's activities. This chemical energy is stored in "
                "carbohydrate molecules, such as sugars and starches, which are synthesized from "
                "carbon dioxide and water. Oxygen is also released as a byproduct. Most plants, "
                "algae, and cyanobacteria perform photosynthesis. Such organisms are called "
                "photoautotrophs."
            ),
            "question": "What is released as a byproduct of photosynthesis?",
            "answer_text": "Oxygen",
            "answer_start": 328,
        },
        {
            "id": "sample_4",
            "title": "Photosynthesis",
            "context": (
                "Photosynthesis is a process used by plants and other organisms to convert light "
                "energy into chemical energy that, through cellular respiration, can later be "
                "released to fuel the organism's activities. This chemical energy is stored in "
                "carbohydrate molecules, such as sugars and starches, which are synthesized from "
                "carbon dioxide and water. Oxygen is also released as a byproduct. Most plants, "
                "algae, and cyanobacteria perform photosynthesis. Such organisms are called "
                "photoautotrophs."
            ),
            "question": "What organisms perform photosynthesis?",
            "answer_text": "plants, algae, and cyanobacteria",
            "answer_start": 383,
        },
        {
            "id": "sample_5",
            "title": "Alan Turing",
            "context": (
                "Alan Mathison Turing OBE FRS was an English mathematician, computer scientist, "
                "logician, cryptanalyst, philosopher, and theoretical biologist. Turing was highly "
                "influential in the development of theoretical computer science, providing a "
                "formalisation of the concepts of algorithm and computation with the Turing machine, "
                "which can be considered a model of a general-purpose computer. Turing is widely "
                "considered to be the father of theoretical computer science and artificial intelligence."
            ),
            "question": "What is Alan Turing considered the father of?",
            "answer_text": "theoretical computer science and artificial intelligence",
            "answer_start": 390,
        },
    ]


def save_predictions(predictions: Dict[str, str], output_path: str) -> None:
    """Save predictions to a JSON file."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)
