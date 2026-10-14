"""Prediction helpers for new single comments and comment batches."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parent
MODEL_FILE = ROOT / "artifacts" / "spam_classifier.joblib"


def load_model():
    if not MODEL_FILE.exists():
        raise FileNotFoundError("Model not found. Run `python train_model.py` first.")
    return joblib.load(MODEL_FILE)


def _spam_score(model, comment: str) -> float:
    """Return a display score; for LinearSVC this is not a calibrated probability."""
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba([comment])[0][list(model.classes_).index(1)])
    margin = float(model.decision_function([comment])[0])
    return 1 / (1 + math.exp(-max(-30, min(30, margin))))


def predict(comment: str, model=None) -> tuple[str, float]:
    """Classify any non-empty input string, not only comments in the training file."""
    if not comment.strip():
        raise ValueError("Please enter a non-empty comment.")
    model = model or load_model()
    label = int(model.predict([comment])[0])
    return ("spam" if label == 1 else "ham"), _spam_score(model, comment)


def predict_many(comments: list[str], model=None) -> list[tuple[str, str, float]]:
    """Classify multiple arbitrary comments; blank lines are ignored."""
    cleaned = [comment.strip() for comment in comments if comment.strip()]
    if not cleaned:
        raise ValueError("Enter at least one non-empty comment.")
    model = model or load_model()
    labels = model.predict(cleaned)
    results = []
    for comment, raw_label in zip(cleaned, labels):
        label = "spam" if int(raw_label) == 1 else "ham"
        results.append((comment, label, _spam_score(model, comment)))
    return results


def main():
    parser = argparse.ArgumentParser(description="Classify any comment as spam or ham.")
    parser.add_argument("comment", nargs="?", help="Any comment text to classify")
    parser.add_argument("--file", help="Optional text file with one comment per line")
    args = parser.parse_args()
    if args.file:
        comments = Path(args.file).read_text(encoding="utf-8").splitlines()
        try:
            results = predict_many(comments)
        except (ValueError, FileNotFoundError) as exc:
            parser.error(str(exc))
        for comment, label, score in results:
            print(f"{label.upper()} | score {score:.1%} | {comment}")
        print("Scores are approximate and not calibrated; review predictions before moderation.")
        return
    comment = args.comment if args.comment is not None else input("Enter any comment: ")
    try:
        label, score = predict(comment)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    print(f"Prediction: {label.upper()}")
    print(f"Spam score (approximate): {score:.1%}")
    print("The score is not calibrated. This educational model can make mistakes; review before moderating.")


if __name__ == "__main__":
    main()
