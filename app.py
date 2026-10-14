"""Interactive single-comment and batch spam detection app."""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from spam_detector import predict, predict_many

ROOT = Path(__file__).resolve().parent
MODEL_FILE = ROOT / "artifacts" / "spam_classifier.joblib"

st.set_page_config(page_title="Spam Comment Detector", page_icon="🛡️", layout="centered")
st.title("Spam Comment Detector")
st.write(
    "Type or paste **any new comment**—it does not need to be in the training dataset. "
    "The machine-learning model predicts **SPAM** or **HAM (not spam)**."
)

if not MODEL_FILE.exists():
    st.warning("No trained model found. Run `python train_model.py` in the project folder first.")
    st.stop()

@st.cache_resource
def load_model(model_path: str, modified_time: float):
    # Include modification time in the cache key so retraining refreshes the deployed model.
    return joblib.load(model_path)

model = load_model(str(MODEL_FILE), MODEL_FILE.stat().st_mtime)
single_tab, batch_tab = st.tabs(["Check one comment", "Check several comments"])

with single_tab:
    comment = st.text_area(
        "Enter any comment", key="single_comment",
        placeholder="Example: Congratulations! Click this link to claim your free prize.",
        height=130,
    )
    if st.button("Detect spam", type="primary", key="detect_one"):
        if not comment.strip():
            st.error("Please type a comment first.")
        else:
            label, score = predict(comment, model)
            if label == "spam":
                st.error("Prediction: SPAM")
            else:
                st.success("Prediction: HAM (not spam)")
            st.progress(score, text=f"Approximate spam score: {score:.1%}")

with batch_tab:
    st.write("Enter one comment per line. Empty lines are skipped.")
    batch_text = st.text_area(
        "Comments", key="batch_comments",
        placeholder="Thanks for the helpful video!\nClick this link to win a free prize.",
        height=180,
    )
    if st.button("Detect all comments", type="primary", key="detect_batch"):
        try:
            results = predict_many(batch_text.splitlines(), model)
            table = pd.DataFrame(
                [{"Comment": text, "Prediction": label.upper(), "Approx. spam score": f"{score:.1%}"}
                 for text, label, score in results]
            )
            st.dataframe(table, use_container_width=True, hide_index=True)
            st.download_button(
                "Download results as CSV", table.to_csv(index=False).encode("utf-8"),
                file_name="spam_detection_results.csv", mime="text/csv",
            )
        except ValueError as exc:
            st.error(str(exc))

st.caption(
    "This is an educational model trained on the UCI YouTube Spam Collection. "
    "It can process new text, but may misclassify comments unlike its training data. "
    "The spam score is not a calibrated probability; review results before moderation."
)
with st.expander("About this project"):
    st.write(
        "The model was selected using group cross-validation and tested on a whole video "
        "excluded from training. Its benchmark score is not a guarantee of accuracy on all "
        "languages, platforms, or future comments."
    )
