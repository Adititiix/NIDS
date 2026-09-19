"""CSV dataset loading for Phase 6 offline training."""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger("nids.ml.dataset")


def load_dataset_csv(path: str, sample_size: int | None = None, random_state: int = 42) -> pd.DataFrame:
    """
    Load a CICFlowMeter-style CSV. Column names are NOT normalized here --
    that's `feature_mapping.normalize_columns()`'s job, kept separate so
    this function stays a pure "read the file" step.

    `sample_size`, if given and smaller than the dataset, draws a random
    (seeded) subsample -- useful for testing the pipeline quickly without
    processing the full multi-gigabyte CIC-IDS2017 release every time.
    """
    logger.info("Loading dataset from %s", path)
    df = pd.read_csv(path, low_memory=False)
    logger.info("Loaded %d rows, %d columns", len(df), len(df.columns))

    if sample_size is not None and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)
        logger.info("Subsampled to %d rows (random_state=%d)", len(df), random_state)

    return df
