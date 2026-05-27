"""
Shared config and data-prep helpers for the MTUM Phase 2 train/score scripts.

"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


# ── Paths ──────────────────────────────────────────────────────────────────
TRAINING_DATA_PATH = Path("Data/covariates_modeling_uplift_models_2026-03-13.csv")
DEPLOYMENT_DATA_PATH = Path("Data/covariates_deployment_dataset_2026-05-16.csv")

MODEL_OUTPUT_PATH = Path("Models/mtum_phase_2_catboost.cbm")
TREATMENT_OUTPUT_PATH = Path("Deployement_dataset/deployement")


# ── Run params ─────────────────────────────────────────────────────────────
TOP_PCT = 0.5  # fraction of scored cohort to write out (1.00 = everyone)
RANDOM_SEED = 42


# ── Treatment encoding ─────────────────────────────────────────────────────
DROP_INDICATORS: List[str] = [
    "BNLX_ChurnP_SKUd_test_export.csv",
    "BNLX_ChurnP_SKUd_controle_export.csv",
    "BNLX_ChurnP_niks_test_export.csv",
    "BNLX_ChurnP_niks_controle_export.csv",
    "BNLX_ChurnP_25_test_export.csv",
    "BNLX_ChurnP_25_controle_export.csv",
]

TREATMENT_CONVERTER: Dict[str, int] = {
    "BNLX_ChurnP_10_test_export.csv": 1,
    "BNLX_ChurnP_10_controle_export.csv": 0,
    "BNLX_ChurnP_25_test_export.csv": 2,
    "BNLX_ChurnP_25_controle_export.csv": 0,
    "BNLX_ChurnP_5eu_test_export.csv": 3,
    "BNLX_ChurnP_5eu_controle_export.csv": 0,
    "BNLX_ChurnP_10eu_test_export.csv": 4,
    "BNLX_ChurnP_10eu_controle_export.csv": 0,
    "BNLX_ChurnP_250_test_export.csv": 5,
    "BNLX_ChurnP_250_controle_export.csv": 0,
    "BNLX_ChurnP_500_test_export.csv": 6,
    "BNLX_ChurnP_500_controle_export.csv": 0,
    "BNLX_ChurnP_SKUe_test_export.csv": 7,
    "BNLX_ChurnP_SKUe_controle_export.csv": 0,
}

K_TO_FILENAME: Dict[int, str] = {
    1: "BNLX_ChurnP_10_test_export.csv",
    3: "BNLX_ChurnP_5eu_test_export.csv",
    4: "BNLX_ChurnP_10eu_test_export.csv",
    5: "BNLX_ChurnP_250_test_export.csv",
    6: "BNLX_ChurnP_500_test_export.csv",
    7: "BNLX_ChurnP_SKUe_test_export.csv",
}
K_VALUES: List[int] = list(K_TO_FILENAME.keys())


# ── Feature schema ─────────────────────────────────────────────────────────
CATEGORICAL_COLS: List[str] = ["has_rfl", "gender", "country_sk"]
NUMERIC_COLS: List[str] = [
    "recency", "frequency", "monetary_value", "total_volume",
    "length_of_relationship", "online_sales", "retail_sales",
    "food_total", "vhms_total", "sports_total", "beauty_total",
    "frequency_52wk", "monetary_value_52wk", "volume_52wk",
    "online_sales_52w", "retail_sales_52w",
    "frequency_53w_104w", "monetary_value_53w_104w",
    "volume_53w_104w", "online_sales_53w_104w", "retail_sales_53w_104w",
]

TARGET_COL = "multi_outcome"


# ── Shared data-prep helpers ───────────────────────────────────────────────
def coerce_metrics_to_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Strip thousands-separators and coerce listed columns to numeric."""
    df = df.copy()
    df[cols] = (
        df[cols]
        .replace({",": ""}, regex=True)
        .apply(pd.to_numeric, errors="coerce")
    )
    return df


def filter_and_encode_treatments(df: pd.DataFrame) -> pd.DataFrame:
    """Drop non-performing incentive arms and map treatment_indicator to int codes."""
    df = df[~df["treatment_indicator"].isin(DROP_INDICATORS)].copy()
    df["treatment"] = df["treatment_indicator"].map(TREATMENT_CONVERTER)
    return df


def cast_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce numeric covariates to int64 and categorical covariates to object."""
    df = coerce_metrics_to_numeric(df, NUMERIC_COLS)
    df[CATEGORICAL_COLS] = df[CATEGORICAL_COLS].astype("object")
    df[NUMERIC_COLS] = df[NUMERIC_COLS].astype("int64")
    return df


def build_multi_outcome_target(df: pd.DataFrame) -> pd.DataFrame:
    """Construct the multi-outcome label: reactivated_<t> / no_reactivated_<t>."""
    df = df.copy()
    df[TARGET_COL] = np.where(
        df["reactivated"].to_numpy() != 0,
        "reactivated_" + df["treatment"].astype(str),
        "no_reactivated_" + df["treatment"].astype(str),
    )
    return df


def fill_cat_missing(df: pd.DataFrame, cat_cols: List[str]) -> pd.DataFrame:
    """Replace NA in categorical columns with the literal 'MISSING'."""
    df = df.copy()
    df[cat_cols] = df[cat_cols].astype("string").fillna("MISSING")
    return df


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Select the training feature columns and apply the categorical NA fill."""
    X = df[CATEGORICAL_COLS + NUMERIC_COLS].copy()
    X = fill_cat_missing(X, CATEGORICAL_COLS)
    return X


def get_treatment_probs_from_y_true(
    df: pd.DataFrame,
    *,
    y_true_col: str = "y_true",
) -> Dict[int, float]:
    """
    Compute P(T=t) using only the last digit of y_true (e.g., 'reactivated_3' -> 3).
    Returns a dict like {0: 0.52, 1: 0.11, ...}.
    """
    probs = (
        df[y_true_col]
        .astype(str)
        .str[-1]
        .astype(int)
        .value_counts(normalize=True)
        .sort_index()
        .to_dict()
    )
    return probs


def load_and_prep_training_df(path: Path = TRAINING_DATA_PATH) -> pd.DataFrame:
    """Load the training CSV and apply the full prep pipeline."""
    df = pd.read_csv(path, low_memory=False)
    df = filter_and_encode_treatments(df)
    df = cast_dtypes(df)
    df = build_multi_outcome_target(df)
    return df