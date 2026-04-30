JAMK Course Similarity Search - POC
=====================================

SETUP
-----
1. Make sure you have these Python packages:
   pip install flask sentence-transformers numpy

2. Place tietojenkasittely.json in the same folder as app.py

RUN
---
   python app.py

Then open browser at:   http://localhost:5050

The first startup takes ~30 seconds while LaBSE encodes the 69 courses.
After that, every search is instant.

HOW IT WORKS
------------
- Type or paste a course description in ANY language (English, Finnish, or mixed)
- LaBSE maps the input into a shared multilingual semantic space
- The system retrieves the top 5 most similar Finnish courses
- Results are colour-coded:
    Green  (>= 0.87)       Equivalent
    Yellow (0.80 - 0.87)   Potential overlap
    Grey   (< 0.80)        No match

Press Enter to search, Shift+Enter for a new line.

POWERED BY
----------
LaBSE (Language-agnostic BERT Sentence Embedding)
Feng et al., 2022 / Google Research
