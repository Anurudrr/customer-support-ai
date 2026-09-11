"""
Configuration for AI Customer Support NLP System.

All configurable constants and paths are defined here.
"""
from pathlib import Path

# ── Project root ──────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent

# ── Data paths ────────────────────────────────────────────────────────────────
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "twitter_conversations.csv"
TRAIN_PATH = DATA_DIR / "processed" / "train.csv"
VAL_PATH = DATA_DIR / "processed" / "val.csv"
TEST_PATH = DATA_DIR / "processed" / "test.csv"
EVAL_PATH = DATA_DIR / "evaluation_set.csv"

# ── Output paths ──────────────────────────────────────────────────────────────
OUTPUTS_DIR = ROOT_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
EVAL_RESULTS_DIR = OUTPUTS_DIR / "evaluation_results"
REPORTS_DIR = ROOT_DIR / "reports"

# ── Model configuration ───────────────────────────────────────────────────────
# Sentence-Transformers model to use for encoding.
# all-MiniLM-L6-v2 is fast, lightweight, and well-suited for semantic similarity.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Path where the fitted classifier artefacts are saved/loaded.
CLASSIFIER_SAVE_DIR = ROOT_DIR / "models" / "intent_classifier"

# ── Classification ────────────────────────────────────────────────────────────
# Number of nearest neighbours to use when voting for an intent.
KNN_K = 5

# Similarity score below this value triggers an escalation.
CONFIDENCE_THRESHOLD = 0.60

# Random seed for reproducibility.
RANDOM_SEED = 42

# ── Intents ───────────────────────────────────────────────────────────────────
INTENTS = [
    "billing_issue",
    "technical_issue",
    "account_access",
    "complaint_angry",
    "complaint_normal",
    "feature_request",
    "positive",
]

# ── Escalation rules ──────────────────────────────────────────────────────────
# Intents that always cause escalation regardless of confidence.
ALWAYS_ESCALATE_INTENTS = ["feature_request"]

# Intents that escalate when combined with detected anger.
ANGRY_ESCALATE_INTENTS = ["billing_issue", "complaint_angry"]

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
