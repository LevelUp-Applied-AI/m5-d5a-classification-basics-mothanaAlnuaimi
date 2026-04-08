"""
Module 5 Week A — Core Skills Drill: Classification & Evaluation Basics

Complete the three functions below. Each function has a docstring
describing its inputs, outputs, and purpose.

Run your work: python drill.py
Test your work: the autograder runs automatically when you open a PR.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def split_data(df, target_col="churned", test_size=0.2, random_state=42):
    """Split a DataFrame into train and test sets with stratification.

    Args:
        df: DataFrame with features and target column.
        target_col: Name of the target column.
        test_size: Fraction of data to use for testing.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    # TODO: Separate features (X) and target (y), then split with stratification
    pass


def compute_classification_metrics(y_true, y_pred):
    """Compute classification metrics from true and predicted labels.

    Args:
        y_true: Array of true labels (0 or 1).
        y_pred: Array of predicted labels (0 or 1).

    Returns:
        Dictionary with keys: 'accuracy', 'precision', 'recall', 'f1'.
        Values are floats.
    """
    # TODO: Compute all four metrics using scikit-learn functions
    pass


def run_cross_validation(X_train, y_train, n_folds=5, random_state=42):
    """Run stratified k-fold cross-validation with LogisticRegression.

    Args:
        X_train: Training features (numeric only).
        y_train: Training labels.
        n_folds: Number of CV folds.
        random_state: Random seed.

    Returns:
        Dictionary with keys: 'scores' (array of fold scores),
        'mean' (float), 'std' (float).
    """
    # TODO: Create a LogisticRegression model and run cross_val_score
    pass


if __name__ == "__main__":
    # Load data
    df = pd.read_csv("data/telecom_churn.csv")
    print(f"Loaded {len(df)} rows")

    # Task 1: Split
    numeric_cols = ["tenure", "monthly_charges", "total_charges",
                    "num_support_calls", "senior_citizen", "has_partner",
                    "has_dependents"]
    df_numeric = df[numeric_cols + ["churned"]]

    result = split_data(df_numeric)
    if result is not None:
        X_train, X_test, y_train, y_test = result
        print(f"Train: {len(X_train)}, Test: {len(X_test)}")

        # Task 2: Metrics
        model = LogisticRegression(random_state=42, max_iter=1000, class_weight="balanced")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = compute_classification_metrics(y_test, y_pred)
        if metrics:
            print(f"Metrics: {metrics}")

        # Task 3: Cross-validation
        cv_results = run_cross_validation(X_train, y_train)
        if cv_results:
            print(f"CV: {cv_results['mean']:.3f} +/- {cv_results['std']:.3f}")
