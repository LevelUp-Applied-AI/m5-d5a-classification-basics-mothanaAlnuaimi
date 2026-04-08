"""Autograder tests for Drill 5A — Classification & Evaluation Basics."""

import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from drill import split_data, compute_classification_metrics, run_cross_validation


@pytest.fixture
def df():
    return pd.read_csv(
        os.path.join(os.path.dirname(__file__), "..", "data", "telecom_churn.csv")
    )


@pytest.fixture
def numeric_df(df):
    cols = ["tenure", "monthly_charges", "total_charges",
            "num_support_calls", "senior_citizen", "has_partner",
            "has_dependents", "churned"]
    return df[cols]


def test_train_test_split_sizes(numeric_df):
    result = split_data(numeric_df)
    assert result is not None, "split_data returned None"
    X_train, X_test, y_train, y_test = result
    total = len(X_train) + len(X_test)
    assert total == len(numeric_df), f"Split lost rows: {total} != {len(numeric_df)}"
    test_ratio = len(X_test) / total
    assert 0.18 <= test_ratio <= 0.22, f"Test ratio {test_ratio:.2f} not ~0.20"


def test_stratification_preserved(numeric_df):
    result = split_data(numeric_df)
    assert result is not None
    X_train, X_test, y_train, y_test = result
    orig_ratio = numeric_df["churned"].mean()
    train_ratio = y_train.mean()
    test_ratio = y_test.mean()
    assert abs(train_ratio - orig_ratio) < 0.03, f"Train churn ratio {train_ratio:.3f} differs from original {orig_ratio:.3f}"
    assert abs(test_ratio - orig_ratio) < 0.03, f"Test churn ratio {test_ratio:.3f} differs from original {orig_ratio:.3f}"


def test_classification_metrics(numeric_df):
    result = split_data(numeric_df)
    assert result is not None
    X_train, X_test, y_train, y_test = result

    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(random_state=42, max_iter=1000, class_weight="balanced")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = compute_classification_metrics(y_test, y_pred)
    assert metrics is not None, "compute_classification_metrics returned None"
    for key in ["accuracy", "precision", "recall", "f1"]:
        assert key in metrics, f"Missing key: {key}"
        assert metrics[key] > 0, f"{key} should be > 0, got {metrics[key]}"


def test_cross_validation_scores(numeric_df):
    result = split_data(numeric_df)
    assert result is not None
    X_train, X_test, y_train, y_test = result

    cv_results = run_cross_validation(X_train, y_train)
    assert cv_results is not None, "run_cross_validation returned None"
    assert "scores" in cv_results, "Missing 'scores' key"
    assert "mean" in cv_results, "Missing 'mean' key"
    assert "std" in cv_results, "Missing 'std' key"
    assert len(cv_results["scores"]) == 5, f"Expected 5 fold scores, got {len(cv_results['scores'])}"
    assert cv_results["mean"] > 0.5, f"Mean CV score {cv_results['mean']:.3f} should be > 0.5"
