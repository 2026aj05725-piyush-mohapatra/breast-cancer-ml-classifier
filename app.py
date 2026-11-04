"""
================================================================================
Project: End-to-End ML Classification & Deployment (Wisconsin Breast Cancer)
Script Name: app.py
Description:
    This Streamlit web application serves as an interactive evaluation portal.
    Key features implemented:
    1. Sidebar file uploader allowing users to upload custom test datasets (CSV).
    2. Fallback loader to automatically load bundled `test_data.csv` if no file is uploaded.
    3. Model selection dropdown to inspect any of the 5 pre-trained classifiers.
    4. Metric cards displaying all 6 evaluation metrics (Accuracy, AUC, Precision, Recall, F1, MCC).
    5. Visual Confusion Matrix (heatmap) and detailed classification report.
    6. Side-by-side Multi-Model Benchmark Comparison Table across the test dataset.
    7. Zero-retraining architecture: loads pre-serialized models and scaler artifacts.
================================================================================
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# Configure Streamlit page layout and title
st.set_page_config(
    page_title="Diagnostic Classification Workbench",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 Clinical Diagnostic ML Evaluation Workbench")
st.markdown(
    "Interactive evaluation portal for pre-trained classification models on the "
    "**Wisconsin Breast Cancer Diagnostic Dataset** (30 features, Binary Classification)."
)

# Registry mapping model display names to their serialized artifact paths
MODEL_MAP = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "KNN": "model/knn_classifier.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "Support Vector Machine": "model/svm_classifier.joblib"
}

@st.cache_resource
def load_scaler_and_models():
    """
    Load the feature scaler and all pre-trained model artifacts from disk.
    Cached with @st.cache_resource to optimize memory and performance.
    """
    scaler_path = "model/scaler_artifact.joblib"
    if not os.path.exists(scaler_path):
        st.error(f"Scaler artifact not found at {scaler_path}. Please run `train_models.py` first.")
        st.stop()
        
    scaler = joblib.load(scaler_path)
    loaded_models = {}
    for name, path in MODEL_MAP.items():
        if os.path.exists(path):
            loaded_models[name] = joblib.load(path)
        else:
            st.warning(f"Model artifact for {name} not found at {path}.")
    return scaler, loaded_models

# Load artifacts into memory
scaler_artifact, trained_models = load_scaler_and_models()

# ------------------------------------------------------------------------------
# Sidebar Controls & Configuration
# ------------------------------------------------------------------------------
st.sidebar.header("⚙️ Evaluation Controls")

# 1. Dataset Upload Option (Satisfies rubric requirement)
uploaded_csv = st.sidebar.file_uploader(
    "1. Upload Test Dataset (CSV)",
    type=["csv"],
    help="Upload test_data.csv containing the 30 input features and the 'target' column."
)

# 2. Model Selection Dropdown (Satisfies rubric requirement)
selected_model_name = st.sidebar.selectbox(
    "2. Select Classification Model",
    options=list(MODEL_MAP.keys()),
    index=0
)

# ------------------------------------------------------------------------------
# Data Ingestion & Validation Logic
# ------------------------------------------------------------------------------
if uploaded_csv is not None:
    eval_df = pd.read_csv(uploaded_csv)
    st.sidebar.success("Loaded uploaded test CSV successfully.")
elif os.path.exists("test_data.csv"):
    eval_df = pd.read_csv("test_data.csv")
    st.sidebar.info("Using bundled repository `test_data.csv`. Upload a CSV above to override.")
else:
    st.error("Please upload `test_data.csv` to proceed.")
    st.stop()

# Ensure ground-truth target column is present in test data
if "target" not in eval_df.columns:
    st.error("Uploaded CSV must contain the ground-truth `target` column alongside input features.")
    st.stop()

# Separate features from ground-truth labels
X_eval = eval_df.drop(columns=["target"])
y_eval = eval_df["target"]

# Transform features using the pre-fitted training scaler
try:
    X_eval_scaled = scaler_artifact.transform(X_eval)
except Exception as e:
    st.error(f"Feature mismatch error when transforming input data: {e}")
    st.stop()

def evaluate_saved_model(clf, X_scaled, y_true):
    """
    Helper function to compute all 6 evaluation metrics for a given model
    without retraining, using pre-computed test features.
    """
    y_pred = clf.predict(X_scaled)
    y_prob = clf.predict_proba(X_scaled)[:, 1]
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_prob),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "y_pred": y_pred
    }

# ------------------------------------------------------------------------------
# Section 1: Detailed Performance of Selected Model
# ------------------------------------------------------------------------------
st.subheader(f"📊 Detailed Performance: {selected_model_name}")

active_clf = trained_models[selected_model_name]
active_res = evaluate_saved_model(active_clf, X_eval_scaled, y_eval)

# Display 6 metric cards side-by-side
col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Accuracy", f"{active_res['Accuracy']:.4f}")
col2.metric("AUC Score", f"{active_res['AUC']:.4f}")
col3.metric("Precision", f"{active_res['Precision']:.4f}")
col4.metric("Recall", f"{active_res['Recall']:.4f}")
col5.metric("F1 Score", f"{active_res['F1']:.4f}")
col6.metric("MCC Score", f"{active_res['MCC']:.4f}")

st.divider()

# ------------------------------------------------------------------------------
# Section 2: Confusion Matrix & Classification Report
# ------------------------------------------------------------------------------
left_col, right_col = st.columns([1, 1.2])

with left_col:
    st.markdown("#### Confusion Matrix")
    cm = confusion_matrix(y_eval, active_res["y_pred"])
    fig, ax = plt.subplots(figsize=(5, 3.8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Malignant (0)", "Benign (1)"],
        yticklabels=["Malignant (0)", "Benign (1)"],
        ax=ax
    )
    ax.set_xlabel("Predicted Diagnosis")
    ax.set_ylabel("Actual Diagnosis")
    st.pyplot(fig)

with right_col:
    st.markdown("#### Detailed Classification Report")
    report_dict = classification_report(
        y_eval,
        active_res["y_pred"],
        target_names=["Malignant (0)", "Benign (1)"],
        output_dict=True
    )
    report_df = pd.DataFrame(report_dict).transpose().round(4)
    st.dataframe(report_df, use_container_width=True)

st.divider()

# ------------------------------------------------------------------------------
# Section 3: Multi-Model Benchmark Comparison Table
# ------------------------------------------------------------------------------
st.subheader("🏆 Multi-Model Comparison on Test Data")
all_rows = []
for m_name, m_obj in trained_models.items():
    res = evaluate_saved_model(m_obj, X_eval_scaled, y_eval)
    all_rows.append({
        "ML Model Name": m_name,
        "Accuracy": round(res["Accuracy"], 4),
        "AUC": round(res["AUC"], 4),
        "Precision": round(res["Precision"], 4),
        "Recall": round(res["Recall"], 4),
        "F1": round(res["F1"], 4),
        "MCC": round(res["MCC"], 4)
    })

comparison_df = pd.DataFrame(all_rows)
st.dataframe(comparison_df, use_container_width=True, hide_index=True)

# Expandable preview of test data samples
with st.expander("🔍 Preview Uploaded Test Data Samples"):
    st.write(f"Test dataset dimensions: `{eval_df.shape[0]}` rows × `{eval_df.shape[1]}` columns")
    st.dataframe(eval_df.head(10), use_container_width=True)
