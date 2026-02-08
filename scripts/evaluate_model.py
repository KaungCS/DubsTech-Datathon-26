from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, top_k_accuracy_score


def _bootstrap_ci(values, alpha=0.05):
    lower = np.quantile(values, alpha / 2)
    upper = np.quantile(values, 1 - alpha / 2)
    half_width = (upper - lower) / 2
    return lower, upper, half_width


def evaluate_model(
    cleaned_path=None,
    model_path=None,
    bootstrap=False,
    n_boot=1000,
    alpha=0.05,
    seed=42,
):
    script_dir = Path(__file__).resolve().parent
    if cleaned_path is None:
        cleaned_path = script_dir / ".." / "data" / "processed" / "cleaned_dataset.csv"
    if model_path is None:
        model_path = script_dir / ".." / "models" / "violation_predictor.pkl"

    test_data = pd.read_csv(cleaned_path)
    feature_cols = [
        "html_file_name",
        "html_file_path",
        "web_URL",
        "domain_category",
        "affected_html_elements",
    ]
    X_test = test_data[feature_cols].copy()
    X_test["text_features"] = (
        X_test[["html_file_name", "html_file_path", "web_URL", "affected_html_elements"]]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
    )
    y_test = test_data["violation_name"]

    bundle = joblib.load(model_path)
    pipeline = bundle["pipeline"]
    frequent_labels = set(bundle.get("frequent_labels", []))
    y_test = y_test.apply(lambda label: label if label in frequent_labels else "Other")

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)
    classes = pipeline.named_steps["model"].classes_

    print("Classification Report for violation_name:")
    print(classification_report(y_test, y_pred, zero_division=0))
    print("Top-3 accuracy:", end=" ")
    print(top_k_accuracy_score(y_test, y_proba, k=3, labels=classes))

    if bootstrap:
        rng = np.random.default_rng(seed)
        y_test_array = np.array(y_test)
        acc_samples = []
        top3_samples = []
        indices = np.arange(len(y_test_array))

        for _ in range(n_boot):
            sample_idx = rng.choice(indices, size=len(indices), replace=True)
            sample_y = y_test_array[sample_idx]
            sample_pred = y_pred[sample_idx]
            acc_samples.append(np.mean(sample_pred == sample_y))

            sample_proba = y_proba[sample_idx]
            top3 = top_k_accuracy_score(
                sample_y,
                sample_proba,
                k=3,
                labels=classes,
            )
            top3_samples.append(top3)

        acc_lower, acc_upper, acc_half_width = _bootstrap_ci(
            np.array(acc_samples),
            alpha=alpha,
        )
        top3_lower, top3_upper, top3_half_width = _bootstrap_ci(
            np.array(top3_samples),
            alpha=alpha,
        )

        print(
            f"Accuracy (bootstrap {int((1 - alpha) * 100)}% CI): "
            f"{np.mean(acc_samples):.3f} +/- {acc_half_width:.3f} "
            f"({acc_lower:.3f}, {acc_upper:.3f})"
        )
        print(
            f"Top-3 accuracy (bootstrap {int((1 - alpha) * 100)}% CI): "
            f"{np.mean(top3_samples):.3f} +/- {top3_half_width:.3f} "
            f"({top3_lower:.3f}, {top3_upper:.3f})"
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate the accessibility violation model."
    )
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        help="Compute bootstrap confidence intervals for accuracy metrics.",
    )
    parser.add_argument(
        "--n-boot",
        type=int,
        default=1000,
        help="Number of bootstrap samples to draw.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Alpha for confidence interval (e.g., 0.05 = 95% CI).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for bootstrap sampling.",
    )
    args = parser.parse_args()

    evaluate_model(
        bootstrap=args.bootstrap,
        n_boot=args.n_boot,
        alpha=args.alpha,
        seed=args.seed,
    )