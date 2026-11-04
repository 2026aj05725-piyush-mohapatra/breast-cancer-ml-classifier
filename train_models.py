"""
================================================================================
Project: End-to-End ML Classification & Deployment (Wisconsin Breast Cancer)
Script Name: train_models.py
Description: 
    This script handles the complete end-to-end model training pipeline:
    1. Loads the Breast Cancer Wisconsin (Diagnostic) Dataset from scikit-learn.
    2. Performs a stratified 75/25 train-test split (random_state=725).
    3. Exports the test partition to `test_data.csv` (features + target column).
    4. Applies Z-score feature standardization (`StandardScaler`) on training data
       and serializes the scaler artifact for downstream evaluation.
    5. Instantiates, trains, and serializes five distinct classifiers:
       - Logistic Regression (L2 regularized)
       - Decision Tree Classifier (Max depth = 5)
       - K-Nearest Neighbors (KNN with distance weighting, k=7)
       - Gaussian Naive Bayes (with var_smoothing)
       - Support Vector Machine (RBF kernel, probability=True)
    6. Evaluates all models on the test set across 6 performance metrics:
       - Accuracy, AUC Score, Precision, Recall, F1 Score, and MCC.
    7. Prints the comparative results table and saves benchmark metrics to CSV.
================================================================================
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

def execute_training_pipeline():
    print("[-] Initializing Data Ingestion & Preprocessing Pipeline...")
    
    # --------------------------------------------------------------------------
    # Step 1: Load Dataset
    # --------------------------------------------------------------------------
    # The Breast Cancer Wisconsin dataset contains 569 instances and 30 continuous
    # numerical features extracted from digitized fine-needle aspirate (FNA) images.
    raw_data = load_breast_cancer()
    feature_names = list(raw_data.feature_names)
    
    df_full = pd.DataFrame(raw_data.data, columns=feature_names)
    df_full["target"] = raw_data.target  # 0 = Malignant, 1 = Benign

    print(f"[+] Dataset successfully loaded: {df_full.shape[0]} instances, {len(feature_names)} features.")

    # --------------------------------------------------------------------------
    # Step 2: Stratified Train-Test Split
    # --------------------------------------------------------------------------
    # We allocate 75% for training and 25% for testing (143 test samples).
    # stratify=y_labels ensures proportional class distribution across splits.
    X_features = df_full[feature_names]
    y_labels = df_full["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y_labels, test_size=0.25, random_state=725, stratify=y_labels
    )

    # --------------------------------------------------------------------------
    # Step 3: Export Test Data for Streamlit Evaluation App
    # --------------------------------------------------------------------------
    # As required by the assignment, test_data.csv contains both features and target.
    test_export_df = X_test.copy()
    test_export_df["target"] = y_test.values
    test_export_df.to_csv("test_data.csv", index=False)
    print(f"[+] Test dataset exported to `test_data.csv` (Shape: {test_export_df.shape})")

    # --------------------------------------------------------------------------
    # Step 4: Feature Scaling & Artifact Serialization
    # --------------------------------------------------------------------------
    # Standardize features using mean=0 and variance=1 based strictly on training data
    # to prevent data leakage. The fitted scaler is saved for inference.
    os.makedirs("model", exist_ok=True)
    feature_scaler = StandardScaler()
    X_train_scaled = feature_scaler.fit_transform(X_train)
    X_test_scaled = feature_scaler.transform(X_test)
    
    scaler_path = "model/scaler_artifact.joblib"
    joblib.dump(feature_scaler, scaler_path)
    print(f"[+] Feature scaler artifact serialized to `{scaler_path}`")

    # --------------------------------------------------------------------------
    # Step 5: Define Classifier Registry
    # --------------------------------------------------------------------------
    # Registry mapping each model name to its scikit-learn instance and output filename.
    classifier_registry = {
        "Logistic Regression": (
            LogisticRegression(C=0.8, max_iter=1000, random_state=725),
            "logistic_regression.joblib"
        ),
        "Decision Tree": (
            DecisionTreeClassifier(max_depth=5, min_samples_split=4, random_state=725),
            "decision_tree.joblib"
        ),
        "KNN": (
            KNeighborsClassifier(n_neighbors=7, weights="distance"),
            "knn_classifier.joblib"
        ),
        "Naive Bayes": (
            GaussianNB(var_smoothing=1e-8),
            "naive_bayes.joblib"
        ),
        "Support Vector Machine": (
            SVC(C=1.2, kernel="rbf", probability=True, random_state=725),
            "svm_classifier.joblib"
        )
    }

    metrics_summary = []

    # --------------------------------------------------------------------------
    # Step 6: Train, Evaluate, and Serialize Each Model
    # --------------------------------------------------------------------------
    print("\n[-] Training and evaluating supervised classifiers...")
    for model_label, (clf_instance, filename) in classifier_registry.items():
        # Fit model on standardized training data
        clf_instance.fit(X_train_scaled, y_train)
        
        # Serialize trained model artifact
        model_filepath = os.path.join("model", filename)
        joblib.dump(clf_instance, model_filepath)

        # Generate predictions and class probabilities on the test set
        y_pred = clf_instance.predict(X_test_scaled)
        y_prob = clf_instance.predict_proba(X_test_scaled)[:, 1]

        # Compute all 6 mandatory evaluation metrics
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)

        metrics_summary.append({
            "ML Model Name": model_label,
            "Accuracy": round(acc, 4),
            "AUC": round(auc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1": round(f1, 4),
            "MCC": round(mcc, 4)
        })
        print(f"    [OK] Trained & Evaluated: {model_label}")

    # --------------------------------------------------------------------------
    # Step 7: Output Benchmark Summary Table
    # --------------------------------------------------------------------------
    results_df = pd.DataFrame(metrics_summary)
    print("\n================ EVALUATION METRICS ON TEST DATA ================")
    print(results_df.to_string(index=False))
    print("=================================================================\n")
    
    # Save benchmark table to CSV inside model folder
    results_df.to_csv("model/benchmark_metrics.csv", index=False)
    print("[+] Training pipeline completed successfully!")

if __name__ == "__main__":
    execute_training_pipeline()
