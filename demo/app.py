import json
import os
from pathlib import Path

import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "tietojenkasittely.json"

MODEL_NAME = os.environ.get("MODEL_NAME", "sentence-transformers/LaBSE")

app = Flask(__name__, static_folder="static")


def normalize_field(value):
    if value is None:
        return ""

    if isinstance(value, list):
        return " ".join(str(item).strip() for item in value if str(item).strip())

    return str(value).strip()


with open(DATA_PATH, "r", encoding="utf-8") as file:
    raw_data = json.load(file)


courses = []

for row in raw_data:
    outcomes = normalize_field(row.get("outcomes"))
    contents = normalize_field(row.get("contents"))
    full_text = f"{outcomes} {contents}".strip()

    if not full_text:
        continue

    courses.append({
        "course_id": row.get("course_id", "Unknown ID"),
        "title": row.get("title", "Untitled course"),
        "credits": row.get("credits", ""),
        "outcomes": outcomes[:500],
        "full_text": full_text,
    })


print(f"Loaded {len(courses)} courses.")
print(f"Loading model: {MODEL_NAME}")

model = SentenceTransformer(MODEL_NAME)

print(f"Encoding {len(courses)} Finnish course descriptions.")

db_texts = [course["full_text"] for course in courses]

db_embeddings = model.encode(
    db_texts,
    normalize_embeddings=True,
    convert_to_numpy=True,
    show_progress_bar=True
)

print("Demo app is ready.")


@app.route("/search", methods=["POST"])
def search():
    payload = request.get_json(silent=True) or {}
    query = str(payload.get("query", "")).strip()

    if not query:
        return jsonify({"error": "Please enter a course description."}), 400

    if len(query) > 8000:
        return jsonify({"error": "The query is too long for this demo."}), 400

    query_embedding = model.encode(
        query,
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    scores = db_embeddings @ query_embedding

    top_n = min(3, len(courses))
    top_indices = np.argsort(scores)[::-1][:top_n]

    results = []

    for rank, index in enumerate(top_indices, start=1):
        course = courses[index]
        score = float(scores[index])

        results.append({
            "rank": rank,
            "course_id": course["course_id"],
            "title": course["title"],
            "credits": course["credits"],
            "outcomes": course["outcomes"],
            "score": round(score, 4),
        })

    return jsonify({
        "query": query,
        "results": results,
        "result_count": len(results)
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "courses": len(courses),
        "model": MODEL_NAME
    })


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5050,
        debug=False
    )