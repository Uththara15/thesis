"""Minimal example client for the similarity service.

Run after starting the service:

    uvicorn app.main:app --reload --port 8000

Then:

    python examples/client.py

The script prints the similarity result for one example bilingual pair
covering software engineering content.
"""

from __future__ import annotations

import json
from urllib import request

URL = "http://localhost:8000/similarity"


def main() -> None:
    payload = {
        "finnish": {
            "outcomes": (
                "Opiskelija ymmärtää ohjelmistosuunnittelun perusteet ja "
                "osaa soveltaa olio-ohjelmoinnin periaatteita."
            ),
            "contents": (
                "Olio-ohjelmointi, luokat, oliot, perintä, kapselointi, "
                "polymorfismi."
            ),
            "assessment": "Tentti, harjoitustyöt ja jatkuva arviointi.",
        },
        "english": {
            "outcomes": (
                "The student understands the basics of software design and "
                "is able to apply object-oriented programming principles."
            ),
            "contents": (
                "Object-oriented programming, classes, objects, inheritance, "
                "encapsulation, polymorphism."
            ),
            "assessment": "Exam, assignments and continuous assessment.",
        },
        "field_configuration": "outcomes_raw",
    }

    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        URL,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )

    with request.urlopen(req) as response:
        data = json.loads(response.read().decode("utf-8"))

    print("Similarity result for one bilingual pair:")
    print(json.dumps(data, indent=2))
    print()
    if data["is_similar"]:
        print(f"Decision: SIMILAR (score {data['cosine_similarity']:.4f} >= "
              f"threshold {data['threshold']:.2f}).")
    else:
        print(f"Decision: NOT SIMILAR (score {data['cosine_similarity']:.4f} "
              f"< threshold {data['threshold']:.2f}).")


if __name__ == "__main__":
    main()
