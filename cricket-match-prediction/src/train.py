"""Model training, baseline benchmarking, and hyperparameter tuning module for Cricket Match Prediction."""

import os
from typing import Tuple, Dict, Any
import pandas as pd
import numpy as np
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

CATEGORICAL_FEATURES = ["team1", "team2", "venue", "toss_decision"]
NUMERIC_FEATURES = [
    "team1_won_toss",
    "team1_batting_first",
    "team1_is_home",
    "team2_is_home",
    "team1_form_5",
    "team2_form_5",
    "team1_form_10",
    "team2_form_10",
    "form_diff_5",
    "team1_overall_win_rate",
    "team2_overall_win_rate",
    "overall_win_rate_diff",
    "h2h_matches",
    "team1_h2h_win_rate",
    "team1_venue_win_rate",
    "team2_venue_win_rate",
    "venue_win_rate_diff",
]


def load_features(features_path: str = None) -> pd.DataFrame:
    """Load engineered features CSV with multi-path resolution."""
    if features_path and os.path.exists(features_path):
        return pd.read_csv(features_path)

    candidate_paths = [
        os.path.abspath("data/processed/features.csv"),
        os.path.abspath("cricket-match-prediction/data/processed/features.csv"),
        os.path.join(os.path.dirname(__file__), "..", "data", "processed", "features.csv"),
    ]

    for p in candidate_paths:
        if os.path.exists(p):
            print(f"Loading features from: {p}")
            return pd.read_csv(p)

    raise FileNotFoundError(f"Could not locate features.csv in: {candidate_paths}")


def chronological_split(
    df: pd.DataFrame, test_size: float = 0.20
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataset chronologically to avoid temporal data leakage."""
    sorted_df = df.copy()
    sorted_df["date"] = pd.to_datetime(sorted_df["date"])
    sorted_df = sorted_df.sort_values(by=["date", "match_id"]).reset_index(drop=True)

    split_idx = int(len(sorted_df) * (1.0 - test_size))
    train_df = sorted_df.iloc[:split_idx].copy()
    test_df = sorted_df.iloc[split_idx:].copy()

    return train_df, test_df


def compute_baselines(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, float]:
    """Compute benchmark baseline heuristic accuracies on the chronological test set."""
    y_train = train_df["team1_won"].values
    y_test = test_df["team1_won"].values

    majority_class = int(np.round(np.mean(y_train)))
    pred_majority = np.full(len(test_df), majority_class)
    acc_majority = float(np.mean(pred_majority == y_test))

    pred_toss = test_df["team1_won_toss"].values
    acc_toss = float(np.mean(pred_toss == y_test))

    pred_career = (test_df["team1_overall_win_rate"].values >= test_df["team2_overall_win_rate"].values).astype(int)
    acc_career = float(np.mean(pred_career == y_test))

    pred_form = (test_df["team1_form_5"].values >= test_df["team2_form_5"].values).astype(int)
    acc_form = float(np.mean(pred_form == y_test))

    pred_h2h = (test_df["team1_h2h_win_rate"].values >= 0.50).astype(int)
    acc_h2h = float(np.mean(pred_h2h == y_test))

    baselines = {
        "Majority Class Baseline": acc_majority,
        "Toss Winner Baseline": acc_toss,
        "Career Win-Rate Baseline": acc_career,
        "Rolling Form (Last 5) Baseline": acc_form,
        "Head-to-Head Advantage Baseline": acc_h2h,
    }

    return baselines


def build_pipeline(classifier=None) -> Pipeline:
    """Build a scikit-learn Pipeline with ColumnTransformer preprocessing and classifier."""
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
            ("num", StandardScaler(), NUMERIC_FEATURES),
        ]
    )

    if classifier is None:
        classifier = RandomForestClassifier(n_estimators=100, random_state=42)

    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

    return pipeline


def train_and_evaluate_rf(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> Tuple[Pipeline, Dict[str, float], np.ndarray]:
    """Train untuned baseline RandomForest pipeline on train split and evaluate on test split."""
    feature_cols = CATEGORICAL_FEATURES + NUMERIC_FEATURES

    X_train = train_df[feature_cols]
    y_train = train_df["team1_won"]
    X_test = test_df[feature_cols]
    y_test = test_df["team1_won"]

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
    }

    cm = confusion_matrix(y_test, y_pred)
    return pipeline, metrics, cm


def tune_hyperparameters(
    train_df: pd.DataFrame, test_df: pd.DataFrame, model_output_path: str = None
) -> Tuple[Pipeline, Dict[str, Any], Dict[str, float], np.ndarray]:
    """Tune RandomForestClassifier using GridSearchCV with TimeSeriesSplit and save the best model.

    Tuned Hyperparameters:
    - n_estimators: Number of trees in the forest
    - max_depth: Maximum tree depth (crucial to prevent memorization / overfitting)
    - min_samples_split: Minimum samples required to split an internal node
    - min_samples_leaf: Minimum samples required at a leaf node (regularization)
    - max_features: Number of features to consider when looking for the best split
    """
    feature_cols = CATEGORICAL_FEATURES + NUMERIC_FEATURES

    X_train = train_df[feature_cols]
    y_train = train_df["team1_won"]
    X_test = test_df[feature_cols]
    y_test = test_df["team1_won"]

    pipeline = build_pipeline()

    param_grid = {
        "classifier__n_estimators": [100, 150],
        "classifier__max_depth": [6, 8, 10],
        "classifier__min_samples_split": [5, 10, 15],
        "classifier__min_samples_leaf": [2, 4, 6],
        "classifier__max_features": ["sqrt", 0.25],
    }

    tscv = TimeSeriesSplit(n_splits=5)

    print("Running GridSearchCV with TimeSeriesSplit (5 temporal splits)...")
    grid_search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=tscv,
        scoring="accuracy",
        n_jobs=-1,
    )

    grid_search.fit(X_train, y_train)

    best_pipeline = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_cv_score = grid_search.best_score_

    # Evaluate best model on test set
    y_pred = best_pipeline.predict(X_test)

    tuned_metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "best_cv_accuracy": float(best_cv_score),
    }

    cm = confusion_matrix(y_test, y_pred)

    # Save final model using joblib to both targets
    save_targets = [
        os.path.abspath("models/cricket_model.pkl"),
        os.path.abspath("cricket-match-prediction/models/cricket_model.pkl"),
        os.path.join(os.path.dirname(__file__), "..", "models", "cricket_model.pkl"),
    ]

    saved = set()
    for target in save_targets:
        norm = os.path.normpath(target)
        if norm not in saved:
            os.makedirs(os.path.dirname(norm), exist_ok=True)
            joblib.dump(best_pipeline, norm)
            saved.add(norm)
            print(f"Saved trained pipeline to: {norm}")

    return best_pipeline, best_params, tuned_metrics, cm


def run_full_training():
    """Execute Step 6 (Baselines), Step 7 (Untuned RF), and Step 8 (Hyperparameter Tuning)."""
    df = load_features()
    train_df, test_df = chronological_split(df, test_size=0.20)

    # Step 6
    baselines = compute_baselines(train_df, test_df)
    toss_baseline = baselines["Toss Winner Baseline"]

    # Step 7: Untuned RF
    print("\n--- Training Untuned Baseline Model ---")
    _, untuned_metrics, untuned_cm = train_and_evaluate_rf(train_df, test_df)

    # Step 8: Tuned RF
    print("\n--- Tuning Hyperparameters with TimeSeriesSplit ---")
    best_model, best_params, tuned_metrics, tuned_cm = tune_hyperparameters(train_df, test_df)

    # Comprehensive Comparison Table
    print("\n" + "=" * 70)
    print("STEP 8 HYPERPARAMETER TUNING & FINAL EVALUATION REPORT")
    print("=" * 70)
    print("Best Hyperparameters Discovered:")
    for k, v in best_params.items():
        param_clean = k.replace("classifier__", "")
        print(f"  • {param_clean:<22}: {v}")
    print(f"  • Best TimeSeriesSplit CV Accuracy: {tuned_metrics['best_cv_accuracy'] * 100:.2f}%")

    print("\n" + "-" * 70)
    print(f"{'Metric':<25} {'Step 6 Toss Baseline':<22} {'Step 7 Untuned RF':<20} {'Step 8 Tuned RF':<15}")
    print("-" * 70)
    print(f"{'Accuracy':<25} {toss_baseline * 100:.2f}%{'':<16} {untuned_metrics['accuracy'] * 100:.2f}%{'':<14} {tuned_metrics['accuracy'] * 100:.2f}%")
    print(f"{'Precision':<25} {'N/A':<22} {untuned_metrics['precision'] * 100:.2f}%{'':<14} {tuned_metrics['precision'] * 100:.2f}%")
    print(f"{'Recall':<25} {'N/A':<22} {untuned_metrics['recall'] * 100:.2f}%{'':<14} {tuned_metrics['recall'] * 100:.2f}%")
    print(f"{'F1 Score':<25} {'N/A':<22} {untuned_metrics['f1'] * 100:.2f}%{'':<14} {tuned_metrics['f1'] * 100:.2f}%")

    print("\nFinal Tuned Confusion Matrix:")
    print("                     Predicted Team2 Won (0)   Predicted Team1 Won (1)")
    print(f"Actual Team2 (0):             {tuned_cm[0][0]:<25} {tuned_cm[0][1]}")
    print(f"Actual Team1 (1):             {tuned_cm[1][0]:<25} {tuned_cm[1][1]}")

    print("\nModel Serialization:")
    print("  Model saved to: models/cricket_model.pkl (joblib)")
    print("=" * 70)

    return best_model, best_params, tuned_metrics


if __name__ == "__main__":
    run_full_training()
