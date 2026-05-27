"""
MTUM Phase 2 — scoring / deployment script.

Loads the trained CatBoost model from disk, recomputes treatment priors from
the training CSV, scores the full deployment cohort, picks the optimal
incentive per customer, and writes a single assignment CSV.

Run:
    python mtum_score.py
"""


from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Tuple, Union

import numpy as np
import pandas as pd

from catboost import CatBoostClassifier

from mtum_config import (
    DEPLOYMENT_DATA_PATH,
    K_TO_FILENAME,
    K_VALUES,
    MODEL_OUTPUT_PATH,
    RANDOM_SEED,
    TARGET_COL,
    TOP_PCT,
    TREATMENT_OUTPUT_PATH,
    cast_dtypes,
    get_treatment_probs_from_y_true,
    load_and_prep_training_df,
    prepare_features,
)


# ── MMOA uplift helpers (scoring-only) ─────────────────────────────────────
def uplift_mmoa(
    df: pd.DataFrame,
    *,
    k: int,
    resp_prefix: str = "reactivated",
    nonresp_prefix: str = "no_reactivated",
    treatment_probs: Dict[int, float],
    return_parts: bool = True,
) -> Union[pd.Series, Tuple[pd.Series, pd.DataFrame]]:
    """
    Modified-outcome uplift estimator for treatment k vs. control (0).

    tau_hat = (p_resp_k / P(T=k) + p_nonresp_0 / P(T=0))
            - (p_nonresp_k / P(T=k) + p_resp_0 / P(T=0))
    """
    r0 = f"p_{resp_prefix}_0"
    rk = f"p_{resp_prefix}_{k}"
    nr0 = f"p_{nonresp_prefix}_0"
    nrk = f"p_{nonresp_prefix}_{k}"

    pt_control = float(treatment_probs.get(0, 0.0))
    pt_treat = float(treatment_probs.get(k, 0.0))

    part_rt_k = df[rk] / pt_treat
    part_nrt_0 = df[nr0] / pt_control
    part_nrt_k = df[nrk] / pt_treat
    part_rt_0 = df[r0] / pt_control

    tau_hat = (part_rt_k + part_nrt_0) - (part_nrt_k + part_rt_0)

    if not return_parts:
        return tau_hat

    parts = pd.DataFrame(
        {
            f"part_rt_{k}": part_rt_k,
            "part_nrt_0": part_nrt_0,
            f"part_nrt_{k}": part_nrt_k,
            "part_rt_0": part_rt_0,
            "pt_control": pt_control,
            f"pt_treat_{k}": pt_treat,
        },
        index=df.index,
    )
    return tau_hat, parts


def add_uplifts(
    df: pd.DataFrame,
    k_values: Iterable[int],
    resp_prefix: str,
    nonresp_prefix: str,
    *,
    treatment_probs: Dict[int, float],
    y_true_col: str = "y_true",
) -> pd.DataFrame:
    """Append a uplift_<k> column per treatment k using the MMOA estimator."""
    df = df.copy()

    for k in k_values:
        required_cols = [
            f"p_{resp_prefix}_0",
            f"p_{resp_prefix}_{k}",
            f"p_{nonresp_prefix}_0",
            f"p_{nonresp_prefix}_{k}",
        ]
        if not all(c in df.columns for c in required_cols):
            continue

        tau_k, _ = uplift_mmoa(
            df,
            k=k,
            resp_prefix=resp_prefix,
            nonresp_prefix=nonresp_prefix,
            treatment_probs=treatment_probs,
            return_parts=True,
        )

        df[f"uplift_{k}"] = tau_k

    return df


# ── Scoring helpers ────────────────────────────────────────────────────────
def load_trained_model(path: Path = MODEL_OUTPUT_PATH) -> CatBoostClassifier:
    """Load a CatBoost model previously saved via .save_model()."""
    model = CatBoostClassifier()
    model.load_model(str(path))
    return model


def load_deployment(path: Path) -> pd.DataFrame:
    """Load the full deployment dataset."""
    return pd.read_csv(path, low_memory=False).reset_index(drop=True)


def score_deployment(model: CatBoostClassifier, X_deploy: pd.DataFrame) -> pd.DataFrame:
    """Return per-class probabilities + argmax prediction + max-prob confidence."""
    proba = model.predict_proba(X_deploy)
    y_pred = model.classes_[np.argmax(proba, axis=1)]
    confidence = np.max(proba, axis=1)

    df_predictions = pd.DataFrame(
        proba,
        index=X_deploy.index,
        columns=[f"p_{c}" for c in model.classes_],
    ).assign(
        y_pred=y_pred,
        confidence=confidence,
    )
    return df_predictions


def attach_uplifts_and_optimal_treatment(
    df_predictions: pd.DataFrame,
    treatment_probs: Dict[int, float],
) -> pd.DataFrame:
    """Add per-treatment uplift columns and pick the optimal treatment per row."""
    df_preds = add_uplifts(
        df_predictions,
        k_values=K_VALUES,
        resp_prefix="reactivated",
        nonresp_prefix="no_reactivated",
        treatment_probs=treatment_probs,
    )

    uplift_cols = [f"uplift_{k}" for k in K_VALUES]
    best_col = df_preds[uplift_cols].idxmax(axis=1)

    df_preds["optimal_lift"] = df_preds[uplift_cols].max(axis=1)
    df_preds["optimal_treatment"] = best_col.str.extract(r"(\d+)$")[0].astype(int)
    return df_preds


def rfm_summary_by_incentive(df: pd.DataFrame) -> pd.DataFrame:
    """Mean and median recency / monetary_value per incentive."""
    rfm_cols = ["recency", "monetary_value"]
    summary = (
        df.groupby("original_incentive_name")[rfm_cols]
        .agg(["mean", "median"])
        .round(2)
    )
    summary["n"] = df.groupby("original_incentive_name").size()
    return summary


def select_top_pct(df: pd.DataFrame, top_pct: float) -> pd.DataFrame:
    """Take the top X fraction of the cohort by optimal_lift (1.0 = everyone)."""
    if top_pct >= 1.0:
        return df
    n_keep = int(len(df) * top_pct)
    return (
        df.sort_values(by="optimal_lift", ascending=False)
        .head(n_keep)
        .reset_index(drop=True)
    )


def write_assignments(df: pd.DataFrame, path: Path) -> None:
    """Write customer_nk + assigned optimal incentive only."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df_out = df[["customer_nk", "original_incentive_name"]]
    df_out.to_csv(path, index=False, sep=";")
    print(f"Assignment rows: {len(df)} -> {path}")


# ── Main ───────────────────────────────────────────────────────────────────
def main() -> None:
    pd.set_option("display.max_columns", None)
    np.random.seed(RANDOM_SEED)

    # 1. Recompute treatment priors from training data
    print("Recomputing treatment priors from training data ...")
    df_train = load_and_prep_training_df()
    treatment_probs = get_treatment_probs_from_y_true(df_train, y_true_col=TARGET_COL)
    print(f"Treatment priors: {treatment_probs}")
    del df_train

    # 2. Load trained model
    print(f"Loading trained model from {MODEL_OUTPUT_PATH} ...")
    model = load_trained_model(MODEL_OUTPUT_PATH)
    print(f"Class order: {model.classes_}")

    # 3. Load + prep deployment cohort (full dataset)
    print("Loading deployment data ...")
    df_deployment = load_deployment(DEPLOYMENT_DATA_PATH)
    df_deployment = cast_dtypes(df_deployment)
    X_deploy = prepare_features(df_deployment)
    print(f"Deployment cohort size: {len(df_deployment)}")

    # 4. Score + uplift + pick optimal treatment per customer
    print("Scoring deployment cohort ...")
    df_predictions = score_deployment(model, X_deploy)

    df_preds_uplift = attach_uplifts_and_optimal_treatment(df_predictions, treatment_probs)

    uplift_cols = [f"uplift_{k}" for k in K_VALUES if f"uplift_{k}" in df_preds_uplift.columns]
    proba_cols = [c for c in df_preds_uplift.columns if c.startswith("p_")]
    for col in uplift_cols + proba_cols:
        df_deployment[col] = df_preds_uplift[col].values

    df_deployment["optimal_treatment"] = df_preds_uplift["optimal_treatment"].values
    df_deployment["optimal_lift"] = df_preds_uplift["optimal_lift"].values
    df_deployment["original_incentive_name"] = df_deployment["optimal_treatment"].map(K_TO_FILENAME)

    # 5. Apply top-pct filter and write single assignment CSV
    df_selected = select_top_pct(df_deployment, top_pct=TOP_PCT)
    write_assignments(df_selected, TREATMENT_OUTPUT_PATH)

    print(f"Mean optimal CATE: {df_selected['optimal_lift'].mean():.6f}")

    print("\nRFM per incentive:")
    print(rfm_summary_by_incentive(df_selected))


if __name__ == "__main__":
    main()