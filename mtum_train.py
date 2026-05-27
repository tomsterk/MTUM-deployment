"""
MTUM Phase 2 — training script.

Loads training covariates, fits a CatBoost MultiClass model with balanced
class weights, and saves the trained model to disk.

Run:
    python mtum_train.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from catboost import CatBoostClassifier
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight

from mtum_config import (
    CATEGORICAL_COLS,
    MODEL_OUTPUT_PATH,
    RANDOM_SEED,
    TARGET_COL,
    get_treatment_probs_from_y_true,
    load_and_prep_training_df,
    prepare_features,
)


# ── Helpers ────────────────────────────────────────────────────────────────
def train_catboost(X: pd.DataFrame, y: pd.Series) -> CatBoostClassifier:
    """Train a balanced-class-weight CatBoost MultiClass on the full dataset."""
    class_labels = np.unique(y)
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=class_labels,
        y=y,
    )

    model = CatBoostClassifier(
        verbose=100,
        #n_estimators = 500,
        random_seed=RANDOM_SEED,
        class_names=list(class_labels),
        class_weights=class_weights,
        loss_function="MultiClass",
        thread_count=-1,
    )
    model.fit(X, y, cat_features=CATEGORICAL_COLS)
    return model


def report_training_fit(model: CatBoostClassifier, X: pd.DataFrame, y: pd.Series) -> None:
    """Print a classification report on training data as a sanity check."""
    proba = model.predict_proba(X)
    y_pred = model.classes_[np.argmax(proba, axis=1)]
    print(classification_report(y, y_pred))


# ── Main ───────────────────────────────────────────────────────────────────
def main() -> None:
    pd.set_option("display.max_columns", None)
    np.random.seed(RANDOM_SEED)

    print("Loading training data ...")
    df = load_and_prep_training_df()

    treatment_probs = get_treatment_probs_from_y_true(df, y_true_col=TARGET_COL)
    print(f"Treatment priors: {treatment_probs}")

    X_train = prepare_features(df)
    y_train = df[TARGET_COL]

    print("Training CatBoost MultiClass ...")
    model = train_catboost(X_train, y_train)
    print(f"Training complete. Class order: {model.classes_}")
    print(f"Feature count: {len(model.feature_names_)}")

    report_training_fit(model, X_train, y_train)

    MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODEL_OUTPUT_PATH))
    print(f"Model saved -> {MODEL_OUTPUT_PATH}")


if __name__ == "__main__":
    main()