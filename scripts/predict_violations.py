from pathlib import Path

import joblib
import pandas as pd

from web_scraper import scrape_metadata
from train_model import train_model


VIOLATION_IMPACTS = {
    "Landmark Navigation": {
        "landmark-one-main": [
            "Screen reader users who expect a main content area",
            "Keyboard users who need to skip to main content",
        ],
        "landmark-complementary-is-top-level": [
            "Screen reader users who navigate by landmarks",
            "Keyboard users using landmark shortcuts",
        ],
        "landmark-unique": [
            "Screen reader users navigating between similar landmarks",
            "Keyboard users relying on unique landmark identification",
        ],
        "region": [
            "Screen reader users who navigate by landmarks",
            "Keyboard-only users relying on landmark shortcuts",
        ],
    },
    "Heading & Structure": {
        "heading-order": [
            "Screen reader users navigating by headings",
            "Keyboard-only users relying on structure",
            "Users with cognitive disabilities who need clear structure",
        ],
        "empty-heading": [
            "Screen reader users navigating by headings",
            "Keyboard users jumping between sections",
        ],
        "page-has-heading-one": [
            "Screen reader users who expect page structure",
            "Keyboard users who need proper content hierarchy",
        ],
    },
    "Button & Control Naming": {
        "button-name": [
            "Screen reader users",
            "Speech-input users who rely on button names",
        ],
        "aria-progressbar-name": [
            "Screen reader users",
            "Users of assistive technologies who need context for progress indicators",
        ],
    },
    "Form & Link Labeling": {
        "link-name": [
            "Screen reader users",
            "Users with cognitive disabilities who need clear link context",
        ],
        "label": [
            "Screen reader users filling forms",
            "Users with cognitive disabilities who need clear labels",
            "Speech-input users who rely on label names",
        ],
    },
    "Images & Alt Text": {
        "image-alt": [
            "Screen reader users",
            "Users who disable images or use text-only browsers",
            "Users with low vision relying on alt text",
        ],
        "image-redundant-alt": [
            "Screen reader users (redundant announcements)",
        ],
    },
    "Color & Contrast": {
        "color-contrast": [
            "Users with low vision",
            "Users with color-vision deficiencies",
            "Users in bright or glare-prone environments",
        ],
        "color-contrast-enhanced": [
            "Users with low vision",
            "Users with color-vision deficiencies",
            "Users in bright or glare-prone environments",
        ],
    },
    "ARIA & Attributes": {
        "aria-allowed-attr": [
            "Screen reader users (ARIA parsing errors)",
            "Users of assistive technologies relying on ARIA",
        ],
        "aria-required-attr": [
            "Screen reader users (missing ARIA relationships)",
            "Users of assistive technologies relying on ARIA",
        ],
        "aria-hidden-focus": [
            "Keyboard-only users who can accidentally focus hidden content",
            "Screen reader users with tab order issues",
        ],
    },
    "ID Management": {
        "duplicate-id": [
            "Screen reader users (confusing or broken references)",
            "Users of assistive technologies that depend on unique IDs",
        ],
        "duplicate-id-active": [
            "Screen reader users (broken interactive element references)",
            "Users of assistive technologies depending on unique IDs",
        ],
    },
    "Page Metadata": {
        "document-title": [
            "Screen reader users",
            "Users of assistive technologies who rely on page titles",
        ],
        "html-has-lang": [
            "Screen reader users (incorrect language pronunciation)",
            "Users who rely on language detection for translations",
        ],
    },
    "Keyboard Navigation": {
        "scrollable-region-focusable": [
            "Keyboard-only users who need to interact with scrollable areas",
            "Users with mobility disabilities relying on keyboard navigation",
        ],
    },
}


def get_violation_impacts(label: str):
    """Get affected users for a violation, searching across all groups."""
    for group, violations in VIOLATION_IMPACTS.items():
        if label in violations:
            return violations[label]
    
    # Fallback for unknown violations
    return [
        "Screen reader users",
        "Keyboard-only users",
        "Users with low vision or cognitive disabilities",
    ]


def get_category_priors(domain_category, data_path, top_k=5):
    try:
        df = pd.read_csv(data_path)
    except Exception:
        return []

    if domain_category in df["domain_category"].unique():
        subset = df[df["domain_category"] == domain_category]
    else:
        subset = df

    counts = subset["violation_name"].value_counts().head(top_k)
    total = counts.sum()
    priors = []
    for name, count in counts.items():
        share = (count / total) if total else 0.0
        priors.append((name, int(count), share))
    return priors


def predict_violations(
    url,
    model_path=None,
    top_k=3,
    min_confidence=0.0,
    debug=False,
    prior_k=5,
):
    script_dir = Path(__file__).resolve().parent
    if model_path is None:
        model_path = script_dir / ".." / "models" / "violation_predictor.pkl"

    if not Path(model_path).exists():
        train_model(
            cleaned_path=script_dir / ".." / "data" / "processed" / "cleaned_dataset.csv",
            model_path=model_path,
            summary_only=True,
        )

    bundle = joblib.load(model_path)
    pipeline = bundle["pipeline"]

    metadata = scrape_metadata(url)
    metadata_df = pd.DataFrame([metadata])
    metadata_df["text_features"] = (
        metadata_df[
            ["html_file_name", "html_file_path", "web_URL", "affected_html_elements"]
        ]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
    )
    if debug:
        print("Scraped metadata:")
        print(metadata_df.to_dict(orient="records")[0])
    proba = pipeline.predict_proba(metadata_df)[0]
    classes = pipeline.named_steps["model"].classes_
    top_indices = proba.argsort()[-top_k:][::-1]
    results = [(classes[i], float(proba[i])) for i in top_indices]
    results = [(label, score) for label, score in results if score >= min_confidence]

    prior_path = script_dir / ".." / "data" / "processed" / "cleaned_dataset.csv"
    priors = get_category_priors(metadata.get("domain_category", ""), prior_path, top_k=prior_k)
    return {"predictions": results, "priors": priors}


if __name__ == "__main__":
    url = "https://example.com"
    predictions = predict_violations(url)
    print(predictions)