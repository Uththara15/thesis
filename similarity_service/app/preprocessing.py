"""Preprocessing for the bilingual similarity service.

This module reproduces the preprocessing of Notebook 1 of the thesis
pipeline EXACTLY. Any divergence from Notebook 1 means the similarity
scores returned by this service no longer correspond to the F1 numbers
reported in the thesis evaluation.

Source: notebooks/01_data_preparation.ipynb in the thesis repository.

The preprocessing steps applied to every input field, in order:
  1. Strip surrounding whitespace.
  2. Remove bullet point characters (U+2022, U+2023, U+25E6, U+2043,
     U+2219).
  3. Remove invisible / zero-width characters (U+200B, U+200C, U+200D,
     U+00AD, U+FEFF).
  4. Unicode NFC normalisation. NFC preserves precomposed Finnish
     letters as single code points that match the multilingual tokenizer
     vocabularies. NFD would split each letter from its diacritic and
     fragment the subword representation.
  5. Collapse runs of whitespace to a single space.

Lemmatisation, when applied, uses spaCy with the same models as Notebook 1:
  Finnish: fi_core_news_sm
  English: en_core_web_sm
NER is disabled because it is not needed for lemmatisation, matching
the notebook.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable

import spacy

from app.config import FieldConfiguration, CONFIG_TO_NUMBER

# Artifact classes - identical to Notebook 1
BULLET_CLASS = r"[\u2022\u2023\u25E6\u2043\u2219]"
INVISIBLE_CLASS = r"[\u200B\u200C\u200D\u00AD\uFEFF]"


# spaCy pipelines are cached after first load because each model is
# roughly 50 MB resident and loading is one-off, not per-request.
_nlp_cache: dict[str, spacy.language.Language] = {}


def load_spacy(language: str) -> spacy.language.Language:
    """Load and cache the spaCy pipeline for the requested language.

    Identical to the model selection in Notebook 1: fi_core_news_sm for
    Finnish, en_core_web_sm for English. Larger Finnish spaCy models do
    not exist at the time of writing, so this symmetric small-model
    choice matches the thesis design.
    """
    if language not in _nlp_cache:
        model_name = {"fi": "fi_core_news_sm", "en": "en_core_web_sm"}[language]
        _nlp_cache[language] = spacy.load(model_name)
    return _nlp_cache[language]


def warm_spacy_models() -> None:
    """Preload both spaCy pipelines.

    Called from the FastAPI lifespan context at startup so that the
    first lemmatised request does not pay the model-load cost.
    """
    load_spacy("fi")
    load_spacy("en")


def clean_text(text: str) -> str:
    """Reproduce Notebook 1 clean_text exactly.

    Returns an empty string for any non-string input, matching the
    notebook behaviour. The order of operations matters: bullets and
    invisibles are removed before NFC normalisation so that the
    normalisation step does not waste work on characters that are about
    to be deleted.
    """
    if not isinstance(text, str):
        return ""
    text = text.strip()
    text = re.sub(BULLET_CLASS, "", text)
    text = re.sub(INVISIBLE_CLASS, "", text)
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def lemmatize_text(text: str, language: str) -> str:
    """Lemmatise a single string using spaCy.

    Mirrors the per-token logic of lemmatize_batch from Notebook 1 with
    NER disabled. Returns an empty string for empty or whitespace-only
    inputs and for empty doc outputs, matching the notebook behaviour.
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    nlp = load_spacy(language)
    doc = nlp(text, disable=["ner"])
    if len(doc) == 0:
        return ""
    return " ".join(token.lemma_ for token in doc if not token.is_space)


def lemmatize_batch(
    texts: Iterable[str], language: str, batch_size: int = 32
) -> list[str]:
    """Reproduce Notebook 1 lemmatize_batch exactly.

    Uses nlp.pipe with disable=['ner'], matching the notebook. Empty or
    non-string inputs become empty strings on the way in; empty docs on
    the way out also return empty strings.
    """
    nlp = load_spacy(language)
    input_texts = [t if isinstance(t, str) else "" for t in texts]
    outputs: list[str] = []
    for doc in nlp.pipe(input_texts, batch_size=batch_size, disable=["ner"]):
        if len(doc) == 0:
            outputs.append("")
        else:
            outputs.append(
                " ".join(token.lemma_ for token in doc if not token.is_space)
            )
    return outputs


def build_configuration(
    outcomes: str,
    contents: str,
    assessment: str,
    configuration: FieldConfiguration,
    language: str,
) -> str:
    """Build one of the six configurations from raw field strings.

    Configurations 1 to 3 are raw concatenations after cleaning.
    Configurations 4 to 6 apply lemmatisation after cleaning before
    concatenation. The field combinations are identical to Notebook 1:

        Config 1 / outcomes_raw                 : outcomes
        Config 2 / outcomes_contents_raw        : outcomes + contents
        Config 3 / full_raw                     : outcomes + contents + assessment
        Config 4 / outcomes_lemmatised          : lemmatised outcomes
        Config 5 / outcomes_contents_lemmatised : lemmatised outcomes + contents
        Config 6 / full_lemmatised              : lemmatised outcomes + contents + assessment

    The single-space separator and per-field cleaning order both match
    Notebook 1.
    """
    cfg_num = CONFIG_TO_NUMBER[configuration]

    outcomes = clean_text(outcomes)
    contents = clean_text(contents)
    assessment = clean_text(assessment)

    if cfg_num in (4, 5, 6):
        outcomes = lemmatize_text(outcomes, language)
        contents = lemmatize_text(contents, language)
        assessment = lemmatize_text(assessment, language)

    if cfg_num in (1, 4):
        return outcomes
    if cfg_num in (2, 5):
        return f"{outcomes} {contents}".strip()
    return f"{outcomes} {contents} {assessment}".strip()
