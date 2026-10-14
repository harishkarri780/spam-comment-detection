"""Train and evaluate spam classifiers on the UCI YouTube Spam Collection."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "uci_youtube_spam"
ARTIFACT_DIR = ROOT / "artifacts"
HOLDOUT_VIDEO = "Youtube05-Shakira.csv"
RANDOM_SEED = 42


def load_comments():
    texts, labels, groups = [], [], []
    files = sorted(DATA_DIR.glob("Youtube*.csv"))
    if len(files) != 5:
        raise FileNotFoundError(
            f"Expected 5 UCI CSV files in {DATA_DIR}; found {len(files)}. "
            "See README.md to download the dataset."
        )
    for file in files:
        with file.open(encoding="latin-1", newline="") as f:
            for row in csv.DictReader(f):
                text = (row.get("CONTENT") or "").strip()
                raw_label = (row.get("CLASS") or "").strip()
                if text and raw_label in {"0", "1"}:
                    texts.append(text)
                    labels.append(int(raw_label))  # UCI: 0=ham, 1=spam
                    groups.append(file.name)
    return np.asarray(texts), np.asarray(labels), np.asarray(groups)


def make_models():
    vectorizer = TfidfVectorizer(
        lowercase=True, strip_accents="unicode", ngram_range=(1, 2),
        min_df=1, max_df=0.98, sublinear_tf=True, stop_words="english"
    )
    return {
        "Multinomial Naive Bayes": Pipeline([
            ("tfidf", vectorizer), ("classifier", MultinomialNB(alpha=0.5))
        ]),
        "Linear Support Vector Machine": Pipeline([
            ("tfidf", TfidfVectorizer(
                lowercase=True, strip_accents="unicode", ngram_range=(1, 2),
                min_df=1, max_df=0.98, sublinear_tf=True, stop_words="english"
            )),
            ("classifier", LinearSVC(class_weight="balanced", random_state=RANDOM_SEED))
        ]),
    }


def main():
    ARTIFACT_DIR.mkdir(exist_ok=True)
    texts, labels, groups = load_comments()
    is_test = groups == HOLDOUT_VIDEO
    x_train, y_train, group_train = texts[~is_test], labels[~is_test], groups[~is_test]
    x_test, y_test = texts[is_test], labels[is_test]
    if not len(x_test):
        raise ValueError(f"Holdout file not found: {HOLDOUT_VIDEO}")

    cv = GroupKFold(n_splits=4)
    models = make_models()
    scores = {}
    for name, model in models.items():
        fold_scores = cross_val_score(
            model, x_train, y_train, groups=group_train, cv=cv,
            scoring="f1_macro", n_jobs=-1
        )
        scores[name] = {"fold_f1_macro": fold_scores.tolist(),
                        "mean_f1_macro": float(fold_scores.mean()),
                        "std_f1_macro": float(fold_scores.std())}
        print(f"{name}: group-CV macro-F1={fold_scores.mean():.4f} "
              f"(+/- {fold_scores.std():.4f})")

    winner = max(scores, key=lambda name: scores[name]["mean_f1_macro"])
    model = models[winner].fit(x_train, y_train)
    predictions = model.predict(x_test)
    metrics = {
        "selected_model": winner,
        "dataset": "UCI YouTube Spam Collection (5 videos)",
        "total_samples": int(len(texts)),
        "train_samples": int(len(x_train)),
        "test_samples": int(len(x_test)),
        "holdout_video": HOLDOUT_VIDEO,
        "deployment_training_samples": int(len(texts)),
        "deployment_model_note": "After the held-out evaluation was recorded, the selected model was refit on all five videos for deployment.",
        "label_mapping": {"0": "ham", "1": "spam"},
        "training_class_counts": {
            "ham": int(np.sum(y_train == 0)), "spam": int(np.sum(y_train == 1))
        },
        "test_class_counts": {
            "ham": int(np.sum(y_test == 0)), "spam": int(np.sum(y_test == 1))
        },
        "group_cross_validation": scores,
        "holdout_metrics": {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision_spam": float(precision_score(y_test, predictions, pos_label=1, zero_division=0)),
            "recall_spam": float(recall_score(y_test, predictions, pos_label=1, zero_division=0)),
            "f1_spam": float(f1_score(y_test, predictions, pos_label=1, zero_division=0)),
            "macro_f1": float(f1_score(y_test, predictions, average="macro")),
        },
        "classification_report": classification_report(
            y_test, predictions, labels=[0, 1], target_names=["ham", "spam"],
            zero_division=0, output_dict=True
        ),
        "confusion_matrix_labels_order": ["ham", "spam"],
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=[0, 1]).tolist(),
    }
    # Keep evaluation honest: predictions above were made before this refit.
    # The saved app model uses all labeled examples to improve coverage for new inputs.
    deployment_model = models[winner].fit(texts, labels)
    joblib.dump(deployment_model, ARTIFACT_DIR / "spam_classifier.joblib")
    with (ARTIFACT_DIR / "evaluation_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with (ARTIFACT_DIR / "confusion_matrix.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["actual \\ predicted", "ham", "spam"])
        writer.writerow(["ham", *metrics["confusion_matrix"][0]])
        writer.writerow(["spam", *metrics["confusion_matrix"][1]])
    report = [
        "# Model Evaluation Results", "",
        f"**Selected model:** {winner}",
        f"**Dataset:** {metrics['dataset']} ({len(texts)} comments)",
        f"**Evaluation:** trained on four video groups; held out {HOLDOUT_VIDEO}", "",
        "**Deployment model:** after the test metrics were recorded, the selected pipeline was retrained on all five videos for classifying new comments.", "",
        "## Holdout test metrics", "",
        "| Metric | Score |", "|---|---:|",
    ]
    for key, value in metrics["holdout_metrics"].items():
        report.append(f"| {key.replace('_', ' ').title()} | {value:.4f} |")
    report += ["", "## Group cross-validation (training videos)", "",
               "| Model | Mean macro-F1 | Standard deviation |", "|---|---:|---:|"]
    for name, result in scores.items():
        report.append(f"| {name} | {result['mean_f1_macro']:.4f} | {result['std_f1_macro']:.4f} |")
    report += ["", "## Confusion matrix", "", "Rows are actual classes; columns are predictions."
               " Class order: ham, spam.", "", "| Actual \\ Predicted | Ham | Spam |",
               "|---|---:|---:|"]
    for label, row in zip(["Ham", "Spam"], metrics["confusion_matrix"]):
        report.append(f"| {label} | {row[0]} | {row[1]} |")
    report += ["", "> Results apply to this small dataset and one held-out video; they are not a guarantee"
               " of performance on other platforms or future comments.", ""]
    (ARTIFACT_DIR / "evaluation_report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"\nSelected: {winner}")
    print(f"Holdout results: {metrics['holdout_metrics']}")
    print(f"Artifacts saved to {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
