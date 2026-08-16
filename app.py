import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="CRM Sales Forecasting",
    page_icon="📈",
    layout="wide",
)

# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
DEFAULT_TEST_FILE = ROOT / "test_data.csv"

# ============================================================
# HEADER
# ============================================================

st.title("📈 CRM Sales Opportunity Forecasting")
st.caption(
    "Comparison of six machine learning classification models "
    "for Closed Won vs Closed Lost opportunity prediction."
)

# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Random Forest": "random_forest.joblib",
    "Gradient Boosting": "gradient_boosting.joblib",
}

# ============================================================
# LOAD METADATA
# ============================================================

metadata_file = MODEL_DIR / "metadata.json"
metadata = {}

if metadata_file.exists():
    try:
        with open(metadata_file, "r", encoding="utf-8") as file:
            metadata = json.load(file)
    except Exception:
        metadata = {}

# ============================================================
# TEST DATA UPLOAD
# ============================================================

st.sidebar.header("📂 Test Data")

uploaded_file = st.sidebar.file_uploader(
    "Upload test_data.csv",
    type=["csv"],
)

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.sidebar.success(f"Uploaded {len(data):,} rows")
else:
    if DEFAULT_TEST_FILE.exists():
        data = pd.read_csv(DEFAULT_TEST_FILE)
        st.sidebar.info("Using bundled test_data.csv")
    else:
        st.error("Please upload test_data.csv.")
        st.stop()

# ============================================================
# DATA VALIDATION
# ============================================================

if "target" not in data.columns:
    st.error(
        "The uploaded CSV must contain a 'target' column "
        "for evaluation."
    )
    st.stop()

if not data["target"].isin([0, 1]).all():
    st.error("The target column must contain only 0 and 1.")
    st.stop()

# ============================================================
# MODEL FEATURES
# ============================================================

if metadata.get("features"):
    FEATURES = [
        column
        for column in metadata["features"]
        if column in data.columns
    ]
else:
    excluded_columns = {
        "target",
        "Predicted_Class",
        "Predicted_Label",
        "Won_Probability",
    }
    FEATURES = [
        column
        for column in data.columns
        if column not in excluded_columns
    ]

if not FEATURES:
    st.error("No model features were found in the uploaded dataset.")
    st.stop()

X = data[FEATURES].copy()
y = data["target"].astype(int)

# ============================================================
# LOAD SAVED MODELS
# ============================================================

@st.cache_resource
def load_models():
    loaded_models = {}

    for model_name, filename in MODEL_FILES.items():
        path = MODEL_DIR / filename

        if not path.exists():
            continue

        try:
            loaded_models[model_name] = joblib.load(path)
        except Exception as error:
            st.warning(
                f"Could not load {model_name}: {error}"
            )

    return loaded_models


models = load_models()

if not models:
    st.error(
        "No saved model files were found in the model folder."
    )
    st.stop()

# ============================================================
# MODEL EVALUATION
# ============================================================

def get_threshold(model_name):
    """Return the validation-selected threshold saved in metadata."""
    if metadata.get("results"):
        for result in metadata["results"]:
            if result.get("ML Model Name") == model_name:
                return float(
                    result.get("Decision Threshold", 0.50)
                )
    return 0.50


def evaluate_model(model_name, model, X_data, y_data):
    probability = model.predict_proba(X_data)[:, 1]
    threshold = get_threshold(model_name)

    prediction = (
        probability >= threshold
    ).astype(int)

    metrics = {
        "ML Model Name": model_name,
        "Accuracy": accuracy_score(y_data, prediction),
        "AUC": roc_auc_score(y_data, probability),
        "Precision": precision_score(
            y_data, prediction, zero_division=0
        ),
        "Recall": recall_score(
            y_data, prediction, zero_division=0
        ),
        "F1": f1_score(
            y_data, prediction, zero_division=0
        ),
        "MCC": matthews_corrcoef(
            y_data, prediction
        ),
        "Decision Threshold": threshold,
    }

    return metrics, prediction, probability


all_results = {}
comparison_rows = []

for model_name, model in models.items():
    try:
        metrics, prediction, probability = evaluate_model(
            model_name,
            model,
            X,
            y,
        )

        all_results[model_name] = {
            "metrics": metrics,
            "prediction": prediction,
            "probability": probability,
        }

        comparison_rows.append(metrics)

    except Exception as error:
        st.error(
            f"Error running {model_name}: {error}"
        )

if not comparison_rows:
    st.error(
        "None of the models could process the uploaded data."
    )
    st.stop()

comparison_df = pd.DataFrame(comparison_rows)

comparison_df = comparison_df.sort_values(
    ["Accuracy", "F1", "AUC"],
    ascending=False,
).reset_index(drop=True)

winner = comparison_df.iloc[0]
winner_name = winner["ML Model Name"]

# ============================================================
# WINNER
# ============================================================

st.success(
    f"🏆 Overall Winner: {winner_name} — "
    f"Test Accuracy: {winner['Accuracy']:.2%}"
)

winner_col1, winner_col2, winner_col3 = st.columns(3)

with winner_col1:
    st.metric(
        "Winner Accuracy",
        f"{winner['Accuracy']:.2%}",
    )

with winner_col2:
    st.metric(
        "Winner F1 Score",
        f"{winner['F1']:.2%}",
    )

with winner_col3:
    st.metric(
        "Winner AUC",
        f"{winner['AUC']:.4f}",
    )

# ============================================================
# DATASET SUMMARY
# ============================================================

st.subheader("📊 Dataset Summary")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Test Instances", f"{len(data):,}")

with col2:
    st.metric("Features", len(FEATURES))

with col3:
    st.metric("Closed Won", int((y == 1).sum()))

with col4:
    st.metric("Closed Lost", int((y == 0).sum()))

# ============================================================
# MAJORITY-CLASS BASELINE
# ============================================================

majority_baseline = max(
    (y == 0).mean(),
    (y == 1).mean(),
)

st.caption(
    f"Majority-class baseline accuracy: "
    f"{majority_baseline:.2%}. "
    f"The winning model improves this baseline by "
    f"{winner['Accuracy'] - majority_baseline:.2%}."
)

# ============================================================
# MODEL COMPARISON
# ============================================================

st.subheader("📋 Model Performance Comparison")

comparison_display = comparison_df[
    [
        "ML Model Name",
        "Accuracy",
        "AUC",
        "Precision",
        "Recall",
        "F1",
        "MCC",
    ]
].copy()

for column in [
    "Accuracy",
    "AUC",
    "Precision",
    "Recall",
    "F1",
    "MCC",
]:
    comparison_display[column] = (
        comparison_display[column].round(4)
    )

st.dataframe(
    comparison_display,
    width="stretch",
    hide_index=True,
)

st.caption(
    "The comparison table evaluates all six models "
    "on the uploaded test dataset."
)

# ============================================================
# MODEL SELECTION
# ============================================================

st.sidebar.header("🤖 Model Analysis")

selected_model = st.sidebar.selectbox(
    "Choose model",
    list(models.keys()),
)

selected_results = all_results[selected_model]
selected_metrics = selected_results["metrics"]
selected_prediction = selected_results["prediction"]
selected_probability = selected_results["probability"]

# ============================================================
# SELECTED MODEL METRICS
# ============================================================

st.subheader(
    f"📌 {selected_model} — Detailed Evaluation"
)

metric1, metric2, metric3 = st.columns(3)
metric4, metric5, metric6 = st.columns(3)

with metric1:
    st.metric(
        "Accuracy",
        f"{selected_metrics['Accuracy']:.2%}",
    )

with metric2:
    st.metric(
        "AUC",
        f"{selected_metrics['AUC']:.4f}",
    )

with metric3:
    st.metric(
        "Precision",
        f"{selected_metrics['Precision']:.2%}",
    )

with metric4:
    st.metric(
        "Recall",
        f"{selected_metrics['Recall']:.2%}",
    )

with metric5:
    st.metric(
        "F1 Score",
        f"{selected_metrics['F1']:.2%}",
    )

with metric6:
    st.metric(
        "MCC",
        f"{selected_metrics['MCC']:.4f}",
    )

# ============================================================
# CONFUSION MATRIX
# ============================================================

st.subheader("🔢 Confusion Matrix")

cm = confusion_matrix(
    y,
    selected_prediction,
)

cm_df = pd.DataFrame(
    cm,
    index=["Actual Lost", "Actual Won"],
    columns=["Predicted Lost", "Predicted Won"],
)

st.dataframe(
    cm_df,
    width="stretch",
)

# ============================================================
# CLASSIFICATION REPORT
# ============================================================

st.subheader("📄 Classification Report")

report = classification_report(
    y,
    selected_prediction,
    target_names=["Closed Lost", "Closed Won"],
    output_dict=True,
    zero_division=0,
)

report_df = pd.DataFrame(report).transpose()

st.dataframe(
    report_df.round(4),
    width="stretch",
)

# ============================================================
# PREDICTION RESULTS
# ============================================================

st.subheader("🔮 Prediction Results")

results_df = data.copy()

results_df["Predicted_Class"] = selected_prediction

results_df["Predicted_Label"] = np.where(
    selected_prediction == 1,
    "Closed Won",
    "Closed Lost",
)

results_df["Won_Probability"] = selected_probability

preferred_columns = [
    "id",
    "account_id",
    "owner_id",
    "amount",
    "target",
    "Predicted_Class",
    "Predicted_Label",
    "Won_Probability",
]

visible_columns = [
    column
    for column in preferred_columns
    if column in results_df.columns
]

remaining_columns = [
    column
    for column in results_df.columns
    if column not in visible_columns
]

results_display = results_df[
    visible_columns + remaining_columns
]

st.dataframe(
    results_display.head(100),
    width="stretch",
    height=400,
)

# ============================================================
# DOWNLOAD RESULTS
# ============================================================

csv_data = results_df.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    label="⬇️ Download Prediction Results",
    data=csv_data,
    file_name=(
        selected_model.lower().replace(" ", "_")
        + "_predictions.csv"
    ),
    mime="text/csv",
)

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "CRM Sales Forecasting | "
    "Machine Learning Classification Assignment"
)
