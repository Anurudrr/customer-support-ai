"""
Data loading utilities.

Handles loading CSV datasets into pandas DataFrames, with basic
validation and logging.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from src.config import TRAIN_PATH, VAL_PATH, TEST_PATH, EVAL_PATH

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = {"message", "intent"}


def load_csv(path: Path, required_cols: Optional[set[str]] = None) -> pd.DataFrame:
    """
    Load a CSV file into a DataFrame, validating required columns.

    Parameters
    ----------
    path : Path
        Path to the CSV file.
    required_cols : set[str], optional
        Column names that must be present.

    Returns
    -------
    pd.DataFrame
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    df = pd.read_csv(path)
    logger.info("Loaded %d rows from %s", len(df), path)

    if required_cols:
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns in {path}: {missing}")

    # Drop rows with null messages
    before = len(df)
    df = df.dropna(subset=["message"])
    df["message"] = df["message"].astype(str).str.strip()
    df = df[df["message"].str.len() > 0]
    after = len(df)
    if before != after:
        logger.warning("Dropped %d rows with null/empty messages.", before - after)

    return df


def load_train() -> pd.DataFrame:
    """Load the training split."""
    return load_csv(TRAIN_PATH, REQUIRED_COLUMNS)


def load_val() -> pd.DataFrame:
    """Load the validation split."""
    return load_csv(VAL_PATH, REQUIRED_COLUMNS)


def load_test() -> pd.DataFrame:
    """Load the test split."""
    return load_csv(TEST_PATH, REQUIRED_COLUMNS)


def load_eval() -> pd.DataFrame:
    """Load the manually-labelled evaluation set."""
    return load_csv(EVAL_PATH, REQUIRED_COLUMNS | {"should_escalate"})


def load_all_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (train, val, test) DataFrames."""
    return load_train(), load_val(), load_test()
