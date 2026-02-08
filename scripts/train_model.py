from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, top_k_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def build_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def collapse_rare_labels(labels, min_count):
    value_counts = labels.value_counts()
    frequent = set(value_counts[value_counts >= min_count].index)
    return labels.apply(lambda label: label if label in frequent else "Other"), frequent


def train_model(
    cleaned_path=None,
    model_path=None,
    summary_only=False,
    min_label_count=20,
):
    script_dir = Path(__file__).resolve().parent
    if cleaned_path is None:
        cleaned_path = script_dir / ".." / "data" / "processed" / "cleaned_dataset.csv"
    if model_path is None:
        model_path = script_dir / ".." / "models" / "violation_predictor.pkl"

    data = pd.read_csv(cleaned_path)
    feature_cols = [
        "html_file_name",
        "html_file_path",
        "web_URL",
        "domain_category",
        "affected_html_elements",
    ]
    target_col = "violation_name"
    X = data[feature_cols].copy()
    X["text_features"] = (
        X[["html_file_name", "html_file_path", "web_URL", "affected_html_elements"]]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
    )
    y = data[target_col]
    y, frequent_labels = collapse_rare_labels(y, min_label_count)

    word_vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.9,
        sublinear_tf=True,
    )
    char_vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 4),
        min_df=3,
        max_df=0.9,
        max_features=5000,
        sublinear_tf=True,
    )

    preprocessor = ColumnTransformer(
        [
            ("word", word_vectorizer, "text_features"),
            ("char", char_vectorizer, "text_features"),
            (
                "cat",
                build_encoder(),
                ["domain_category", "affected_html_elements"],
            ),
        ]
    )
    base_model = LinearSVC(class_weight="balanced", random_state=42)
    calibrated_model = CalibratedClassifierCV(
        base_model,
        method="sigmoid",
        cv=3,
    )
    pipeline = Pipeline(
        [
            ("preprocess", preprocessor),
            ("model", calibrated_model),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    pipeline.fit(X_train, y_train)

    y_pred_train = pipeline.predict(X_train)
    y_pred_test = pipeline.predict(X_test)

    y_proba_train = pipeline.predict_proba(X_train)
    y_proba_test = pipeline.predict_proba(X_test)
    classes = pipeline.named_steps["model"].classes_

    if summary_only:
        print("Train metrics summary:")
        report = classification_report(
            y_train, y_pred_train, output_dict=True, zero_division=0
        )
        train_top3 = top_k_accuracy_score(
            y_train, y_proba_train, k=3, labels=classes
        )
        print(
            f"{target_col}: acc={report['accuracy']:.3f} "
            f"macro_f1={report['macro avg']['f1-score']:.3f} "
            f"weighted_f1={report['weighted avg']['f1-score']:.3f} "
            f"top3_acc={train_top3:.3f}"
        )

        print("Test metrics summary:")
        report = classification_report(
            y_test, y_pred_test, output_dict=True, zero_division=0
        )
        test_top3 = top_k_accuracy_score(
            y_test, y_proba_test, k=3, labels=classes
        )
        print(
            f"{target_col}: acc={report['accuracy']:.3f} "
            f"macro_f1={report['macro avg']['f1-score']:.3f} "
            f"weighted_f1={report['weighted avg']['f1-score']:.3f} "
            f"top3_acc={test_top3:.3f}"
        )
    else:
        print("Train metrics:")
        print(f"Classification Report for {target_col}:")
        print(classification_report(y_train, y_pred_train, zero_division=0))
        print("Top-3 accuracy:", end=" ")
        print(top_k_accuracy_score(y_train, y_proba_train, k=3, labels=classes))

        print("Test metrics:")
        print(f"Classification Report for {target_col}:")
        print(classification_report(y_test, y_pred_test, zero_division=0))
        print("Top-3 accuracy:", end=" ")
        print(top_k_accuracy_score(y_test, y_proba_test, k=3, labels=classes))

    joblib.dump(
        {
            "pipeline": pipeline,
            "target_col": target_col,
            "frequent_labels": sorted(frequent_labels),
            "min_label_count": min_label_count,
        },
        model_path,
        compress=9,
    )

    print("Model and encoder saved successfully!")
    return pipeline, (X_test, y_test)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train the accessibility violation model."
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Print only accuracy and F1 summaries for train/test.",
    )
    parser.add_argument(
        "--min-label-count",
        type=int,
        default=20,
        help="Minimum label frequency to keep; others map to 'Other'.",
    )
    args = parser.parse_args()

    train_model(
        summary_only=args.summary_only,
        min_label_count=args.min_label_count,
    )