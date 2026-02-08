# Project Name
iShowAccessibility

# Team Members
Becky Jiang, Justin Nguyen, Michelle Nguyen, Kaung Lin

# Overview
This project predicts the type of accessibility violation based on HTML code snippets and related metadata. 
The goal is to help developers quickly identify and categorize accessibility issues using machine learning.

Web accessibility tools detect violations such as:
- Missing alt text
- Low color contrast
- Duplicate landmarks
- Missing labels

Our goal is to build a classification model that predicts the violation type directly from HTML or descriptive text.

# Approach
1. Data preprocessing
- Combined relevant text columns into a single input string.
- Cleaned and normalized text.
- Converted HTML into model-friendly text features.

2. Feature extraction
- Applied word and character level TF-IDF vectorization to convert text into numerical features.

3. Model
- Trained a calibrated Linear Regression Classifier for multi-class prediction.
- Chosen for strong performance in text classifcation and reliable probability estimates.


## Accessibility Violation Predictor

# Overview
This project predicts the type of web accessibility violation from HTML-related metadata. The model helps developers triage likely issues such as missing alt text, low color contrast, duplicate landmarks, and missing labels.

# Project Structure
- `data/raw/Access_to_Tech_Dataset.csv`: Raw dataset input.
- `data/processed/cleaned_dataset.csv`: Cleaned dataset created by preprocessing.
- `models/violation_predictor.pkl`: Trained model artifact (generated locally).
- `scripts/`: Training, evaluation, and prediction scripts.

# Setup
1. Create and activate a virtual environment (optional but recommended).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

# Train the Model
This pipeline preprocesses the dataset, trains the model, and evaluates it:

```bash
python scripts/main.py --steps preprocess train evaluate
```

Notes:
- Preprocessing reads `data/raw/Access_to_Tech_Dataset.csv` and writes `data/processed/cleaned_dataset.csv`.
- Training writes `models/violation_predictor.pkl`.

Optional training flags:

```bash
python scripts/main.py --steps preprocess train evaluate --summary-only --min-label-count 20
```

# Predict Violations for a URL
After training, you can predict likely violations for a URL:

```bash
python scripts/main.py --steps predict --predict-url "https://example.com" --top-k 3 --min-confidence 0.0
```

Output includes:
- Top predicted violation labels with probabilities
- Affected user groups for each predicted issue
- Category-based prior issues from the dataset

Optional prediction flags:
- `--debug-scrape`: Print the scraped metadata used for prediction.
- `--prior-k`: How many category priors to show.

# Model Approach
1. Data preprocessing: filter valid rows and remove failed scrapes.
2. Feature extraction: word- and character-level TF-IDF over combined text fields.
3. Model: calibrated Linear SVM for multi-class prediction.

# Evaluation Metrics
Bootstrap confidence intervals (95% CI):
- Accuracy: 0.886 +/- 0.011 (0.875, 0.896)
- Top-3 accuracy: 0.977 +/- 0.005 (0.972, 0.982)

Reproduce:
```bash
python scripts/evaluate_model.py --bootstrap --n-boot 1000 --alpha 0.05
```
