"""
mtum_evaluate.py — hold-out evaluation of the MTUM Phase 2 multi-arm uplift model.

Pipeline:
  1. Load training data via mtum_config.load_and_prep_training_df (restricts to
     the kept arms and builds the multi_outcome target).
  2. Stratified 70/30 split on (treatment × reactivation) so both splits keep
     the same arm balance and reactivation rate.
  3. Fit the CatBoost MultiClass model on the 70% train split using the same
     train_catboost helper as the production script.
  4. Score the 30% test split, recover MMOA uplift per arm, pick optimal_lift
     and optimal_treatment per customer (same logic as mtum_score.py).
  4b. Report multiclass log loss + macro-averaged log loss on the test set
      (probabilistic calibration of the underlying MultiClass classifier,
      independent of the uplift framing).
  5. Filter the test set to "policy-matched" rows only: keep all controls and
     the treated rows whose assigned arm == the model's recommended arm.
     Without this filter the Qini is diluted across arm mismatches.
  6. Sort the filtered set by optimal_lift, compute the per-decile uplift
     table (treated vs. control reactivation rate, treated = arm matches
     recommendation).
  7. Render the Qini curve, print AUUC, save the curve as HTML.

Run:
    python mtum_evaluate.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

from mtum_config import (
    RANDOM_SEED,
    TARGET_COL,
    get_treatment_probs_from_y_true,
    load_and_prep_training_df,
    prepare_features,
)
from mtum_train import train_catboost
from mtum_score import (
    attach_uplifts_and_optimal_treatment,
    score_deployment,
)


# ── Decile uplift + Qini helpers (inlined) ────────────────────────────────
def uplift_by_decile_bin(
    df,
    treatment_col="treatment",
    outcome_col="reactivated",
    size=10,
    binary_uplift=True,
):
    """
    Compute per-decile and cumulative uplift metrics from a scored DataFrame.

    Expects `df` to be pre-sorted by predicted uplift (descending), so that
    bin 1 contains the highest-scoring customers and bin `size` the lowest.
    """
    n = len(df)
    bins = range(1, size + 1)
    results = []

    for b in bins:
        start = int(np.ceil(n * (b - 1) / size))
        end = int(np.ceil(n * b / size))
        subset = df.iloc[start:end]

        if binary_uplift:
            t = subset[treatment_col].astype(str).str.strip()
            t_lower = t.str.lower()
            is_control = t_lower.str.startswith("control")
            last_digit = t_lower.str.extract(r"(\d)\s*$", expand=False)
            is_treated = (~is_control) & last_digit.notna()
        else:
            t = subset[treatment_col]
            is_control = t.eq(0)
            is_treated = t.ne(0) & t.notna()

        treated_n = int(is_treated.sum())
        control_n = int(is_control.sum())
        treated_converted_n = int(subset.loc[is_treated, outcome_col].sum())
        control_converted_n = int(subset.loc[is_control, outcome_col].sum())
        treated_rate = float(subset.loc[is_treated, outcome_col].mean()) if treated_n > 0 else 0.0
        control_rate = float(subset.loc[is_control, outcome_col].mean()) if control_n > 0 else 0.0

        results.append(
            {
                "bin": b,
                "bin_start_idx": start,
                "bin_end_idx": end,
                "bin_n": len(subset),
                "treated_n": treated_n,
                "control_n": control_n,
                "treated_converted_n": treated_converted_n,
                "control_converted_n": control_converted_n,
                "treated_rate": treated_rate,
                "control_rate": control_rate,
                "uplift": treated_rate - control_rate,
            }
        )

    df_out = pd.DataFrame(results).sort_values("bin").reset_index(drop=True)

    df_out["cum_treated_n"] = df_out["treated_n"].cumsum()
    df_out["cum_control_n"] = df_out["control_n"].cumsum()
    df_out["cum_treated_converted_n"] = df_out["treated_converted_n"].cumsum()
    df_out["cum_control_converted_n"] = df_out["control_converted_n"].cumsum()

    df_out["cum_treated_rate"] = (
        df_out["cum_treated_converted_n"] / df_out["cum_treated_n"].replace(0, np.nan)
    ).fillna(0.0)
    df_out["cum_control_rate"] = (
        df_out["cum_control_converted_n"] / df_out["cum_control_n"].replace(0, np.nan)
    ).fillna(0.0)

    df_out["cum_population_frac"] = df_out["bin_n"].cumsum() / df_out["bin_n"].sum()

    df_out["inc_gains"] = (
        (df_out["cum_treated_rate"] - df_out["cum_control_rate"])
        * df_out["cum_population_frac"]
    )

    df_out["random_expected"] = df_out["cum_population_frac"] * df_out["inc_gains"].iloc[-1]
    df_out["lift_over_random"] = df_out["inc_gains"] - df_out["random_expected"]

    return df_out


def calc_auuc(df):
    """Area between uplift curve and random baseline (trapezoid rule)."""
    x = np.concatenate([[0], df["cum_population_frac"].values])
    y = np.concatenate([[0], df["lift_over_random"].values])
    return np.trapezoid(y, x)


def plot_incremental_response_rate(uplift_curve_df):
    df = uplift_curve_df.copy()
    df["pct_targeted"] = df["bin"] / df["bin"].max()
    final_inc_gain = df["inc_gains"].iloc[-1]
    auuc = calc_auuc(df)

    fig = px.line(
        df,
        x="pct_targeted",
        y="inc_gains",
        markers=True,
        labels={
            "pct_targeted": "% Targeted",
            "inc_gains": "Cumulative Incremental Gain",
        },
        title="Qini Curve",
    )

    fig.data[0].name = "Model Uplift"
    fig.data[0].showlegend = True

    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, final_inc_gain],
            mode="lines",
            name="Random Targeting",
            line=dict(dash="dash"),
        )
    )

    fig.add_annotation(
        x=0.95, y=1.2,
        xref="paper", yref="paper",
        text=f"AUUC = {auuc:.5f}",
        showarrow=False,
        font=dict(size=13),
        bgcolor="white",
        bordercolor="black",
        borderwidth=1,
        borderpad=4,
    )

    fig.update_layout(
        template="plotly_white",
        title_x=0.5,
        legend_title_text="",
        xaxis=dict(tickformat=".0%"),
        yaxis=dict(tickformat=".2%"),
    )

    return fig


# ── Main ───────────────────────────────────────────────────────────────────
def main() -> None:
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 160)
    np.random.seed(RANDOM_SEED)

    # 1. Load + prep training data (multi-arm encoding, multi_outcome target)
    print("Loading training data ...")
    df = load_and_prep_training_df()
    df = df.reset_index(drop=True)
    print(f"Total rows: {len(df):,}")
    print("\nCounts per treatment arm:")
    print(df["treatment"].value_counts().sort_index().to_string())

    # 2. Stratified 70/30 split on (treatment × reactivation)
    strata = df["treatment"].astype(str) + "_" + df["reactivated"].astype(int).astype(str)
    df_train, df_test = train_test_split(
        df,
        test_size=0.30,
        stratify=strata,
        random_state=RANDOM_SEED,
    )
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    print(f"\nTrain: {len(df_train):,}   Test: {len(df_test):,}")
    print("\nTrain counts per arm:")
    print(df_train["treatment"].value_counts().sort_index().to_string())
    print("\nTest counts per arm:")
    print(df_test["treatment"].value_counts().sort_index().to_string())
    print("\nReactivation rate per arm (test):")
    print(df_test.groupby("treatment")["reactivated"].mean().round(4).to_string())

    # 3. Train CatBoost MultiClass on the 70% train split
    print("\nTraining CatBoost MultiClass on 70% train split ...")
    X_train = prepare_features(df_train)
    y_train = df_train[TARGET_COL]
    model = train_catboost(X_train, y_train)
    print(f"Class order: {model.classes_}")

    # 4. Recompute treatment priors from the TRAINING split (matches what the
    #    model saw — required for the MMOA division by P(T=k))
    treatment_probs = get_treatment_probs_from_y_true(df_train, y_true_col=TARGET_COL)
    print(f"\nTreatment priors (train): {treatment_probs}")

    # 5. Score test set + recover optimal_lift per customer
    print("Scoring 30% test split ...")
    X_test = prepare_features(df_test)
    df_predictions = score_deployment(model, X_test)
    df_preds_uplift = attach_uplifts_and_optimal_treatment(df_predictions, treatment_probs)

    df_test["optimal_lift"] = df_preds_uplift["optimal_lift"].values
    df_test["optimal_treatment"] = df_preds_uplift["optimal_treatment"].values

    print(f"Mean test-set optimal_lift: {df_test['optimal_lift'].mean():+.5f}")
    print(f"Std  test-set optimal_lift: {df_test['optimal_lift'].std():.5f}")

    print("\nOptimal-treatment distribution on test:")
    print(df_test["optimal_treatment"].value_counts().sort_index().to_string())

    # 5b. Multiclass log loss + macro-averaged log loss
    #   - Sample-weighted multiclass log loss: standard sklearn calc. Lower =
    #     better-calibrated probabilities. Implicitly weighted by class
    #     frequency, so the dominant "no_reactivated_*" classes drive the score.
    #   - Macro-averaged log loss: per-class neg-log-prob of the true label,
    #     averaged with equal weight across the 12 outcome classes. Reveals
    #     calibration on the rare "reactivated_*" classes that the
    #     sample-weighted version masks.
    print("\n-- Multiclass log loss --------------------------------------")
    y_true_test = df_test[TARGET_COL].astype(str).to_numpy()
    class_labels = list(model.classes_)
    proba_cols = [f"p_{c}" for c in class_labels]
    proba_test = df_predictions[proba_cols].to_numpy()

    mc_logloss = log_loss(y_true_test, proba_test, labels=class_labels)
    print(f"Multiclass log loss (sample-weighted): {mc_logloss:.5f}")

    eps = 1e-15
    per_class_logloss = {}
    for c in class_labels:
        mask = y_true_test == c
        if not mask.any():
            continue
        p_true = proba_test[mask, class_labels.index(c)]
        p_true_clipped = np.clip(p_true, eps, 1.0 - eps)
        per_class_logloss[c] = float(-np.mean(np.log(p_true_clipped)))

    macro_logloss = float(np.mean(list(per_class_logloss.values())))
    print(f"Macro-averaged log loss:               {macro_logloss:.5f}")
    print("\nPer-class log loss (-mean log p_true within each class):")
    for c in sorted(per_class_logloss):
        n_c = int((y_true_test == c).sum())
        print(f"  {c:30s} {per_class_logloss[c]:.5f}  (n={n_c:,})")

    # 6. Filter to rows where we can honestly evaluate the model's policy:
    #    keep all controls (baseline) AND treated rows whose actually-assigned
    #    arm matches the model's recommended arm. Drop everyone else — we don't
    #    observe what would have happened under the recommended arm.
    n_before = len(df_test)
    df_test_eval = df_test[
        (df_test["treatment"] == 0)
        | (df_test["treatment"] == df_test["optimal_treatment"])
    ].copy()
    n_after = len(df_test_eval)
    print(f"\nPolicy-match filter: {n_before:,} -> {n_after:,} rows kept "
          f"({n_after / n_before * 100:.1f}%)")
    print("After filter, counts per arm:")
    print(df_test_eval["treatment"].value_counts().sort_index().to_string())

    # 7. Sort by optimal_lift desc, run decile uplift in MTUM mode
    #    (treatment column is integer-coded: 0 = control, ≠0 = treated)
    df_ranked = df_test_eval.sort_values("optimal_lift", ascending=False).reset_index(drop=True)
    qini_bins = uplift_by_decile_bin(
        df_ranked,
        treatment_col="treatment",
        outcome_col="reactivated",
        size=10,
        binary_uplift=False,
    )

    print("\nDecile uplift table (test set, treated-vs-control regardless of arm):")
    display_cols = [
        "bin", "bin_n", "treated_n", "control_n",
        "treated_rate", "control_rate", "uplift",
        "cum_treated_rate", "cum_control_rate",
        "inc_gains", "random_expected", "lift_over_random",
    ]
    print(qini_bins[display_cols].round(5).to_string(index=False))

    # 7. Qini curve + AUUC
    auuc = calc_auuc(qini_bins)
    print(f"\nAUUC: {auuc:.5f}")
    print(f"Top-decile empirical uplift:    {qini_bins.loc[0, 'uplift']:+.5f}")
    print(f"Bottom-decile empirical uplift: {qini_bins.loc[9, 'uplift']:+.5f}")
    print(f"Top-vs-bottom spread:           {qini_bins.loc[0, 'uplift'] - qini_bins.loc[9, 'uplift']:+.5f}")

    fig = plot_incremental_response_rate(qini_bins)
    fig.update_layout(
        title="Qini curve — MTUM multi-arm (CatBoost MultiClass + MMOA, 70/30 hold-out test)"
    )

    out_path = Path("Output/qini_curve_evaluation.html")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path))
    print(f"\nQini curve saved -> {out_path}")

    try:
        fig.show()
    except Exception:
        pass


if __name__ == "__main__":
    main()
