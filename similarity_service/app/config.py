"""Service configuration.

Centralises the model identifier, sequence length, configuration list,
and the operating thresholds calibrated on the ICT dataset.

The threshold values are PLACEHOLDERS based on the thesis pattern (LaBSE
sits near the top of a flat F1 plateau, with CV-mean thresholds typically
in the 0.70 to 0.85 range for this task). They must be replaced with the
actual transfer_threshold values from
notebooks/04_generalization_validation.ipynb -> ict_reference[config_num]
['transfer_threshold'] BEFORE this service is deployed against real
data. The placeholder values let the service run for prototype review
without unblocking on the exact numbers.
"""

from __future__ import annotations

from enum import Enum

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

# Matches the Hugging Face identifier used in Notebook 2.
MODEL_NAME: str = "sentence-transformers/LaBSE"

# Matches MAX_SEQ_LEN in Notebook 2.
MAX_SEQ_LENGTH: int = 512

# Matches BATCH_SIZE in Notebook 2. The service rarely uses batching
# in practice (one pair per request), but the value is kept identical
# for embedding-time parity with the notebook pipeline.
BATCH_SIZE: int = 32

# Expected output dimensionality. The notebook asserts this on every
# generated array. The service uses it as a sanity check at startup.
EMBEDDING_DIM: int = 768


# ---------------------------------------------------------------------------
# Configurations
# ---------------------------------------------------------------------------

class FieldConfiguration(str, Enum):
    """The six text configurations evaluated in the thesis.

    The names follow Notebook 1's config_1 to config_6 design exactly.
    Configurations 4 to 6 are the lemmatized counterparts of 1 to 3 using
    identical field combinations.
    """

    outcomes_raw = "outcomes_raw"  # config 1
    outcomes_contents_raw = "outcomes_contents_raw"  # config 2
    full_raw = "full_raw"  # config 3
    outcomes_lemmatised = "outcomes_lemmatised"  # config 4
    outcomes_contents_lemmatised = "outcomes_contents_lemmatised"  # config 5
    full_lemmatised = "full_lemmatised"  # config 6


# Mapping from public configuration name to notebook config number (1..6).
# Used by preprocessing.build_configuration.
CONFIG_TO_NUMBER: dict[FieldConfiguration, int] = {
    FieldConfiguration.outcomes_raw: 1,
    FieldConfiguration.outcomes_contents_raw: 2,
    FieldConfiguration.full_raw: 3,
    FieldConfiguration.outcomes_lemmatised: 4,
    FieldConfiguration.outcomes_contents_lemmatised: 5,
    FieldConfiguration.full_lemmatised: 6,
}


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

# PLACEHOLDER thresholds. Replace from Notebook 4 ict_reference output.
# Pattern from the thesis: LaBSE plateaus are broad, CV-mean thresholds
# are typically in the 0.70 to 0.85 region. Conservative midpoint used
# here so that the service runs end-to-end for review.
OPERATING_THRESHOLDS: dict[FieldConfiguration, float] = {
    FieldConfiguration.outcomes_raw: 0.78,
    FieldConfiguration.outcomes_contents_raw: 0.78,
    FieldConfiguration.full_raw: 0.78,
    FieldConfiguration.outcomes_lemmatised: 0.78,
    FieldConfiguration.outcomes_contents_lemmatised: 0.78,
    FieldConfiguration.full_lemmatised: 0.78,
}

# Default configuration when the request does not specify one. Chosen as
# outcomes_raw because the thesis Notebook 3 reports that LaBSE on raw
# outcomes is the cleanest operating point: short inputs (no truncation),
# raw text (no preprocessing fragility), and on the top F1 plateau.
DEFAULT_CONFIGURATION: FieldConfiguration = FieldConfiguration.outcomes_raw


# ---------------------------------------------------------------------------
# Service metadata
# ---------------------------------------------------------------------------

SERVICE_NAME: str = "bilingual-course-similarity"
SERVICE_VERSION: str = "0.1.0"

# Maximum byte length per text field. Defensive limit, well above any
# realistic Peppi course description, well below pathological payloads.
MAX_TEXT_LENGTH: int = 20_000

# Minimum length to consider a field non-empty.
MIN_TEXT_LENGTH: int = 1
