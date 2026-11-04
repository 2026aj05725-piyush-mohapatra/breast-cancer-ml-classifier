# End-to-End ML Classification & Deployment: Wisconsin Breast Cancer Diagnostics

## 1. Mandatory Submission Links
- **GitHub Repository Link:** `https://github.com/<your-username>/breast-cancer-ml-classifier`
- **Live Streamlit App Link:** `https://<your-app-name>.streamlit.app`

---

## 2. README Structure (Step 5 Deliverable)

### a. Problem Statement
Accurate and early differentiation between malignant and benign breast masses is a critical clinical classification problem. Using digitized fine-needle aspirate (FNA) cell nuclei characteristics, the objective is to build, evaluate, and deploy five supervised machine learning classifiers that predict whether a diagnostic sample is **Malignant (0)** or **Benign (1)** and compare their discriminative performance across six clinical and statistical metrics.

### b. Dataset Description
- **Dataset Name:** Breast Cancer Wisconsin (Diagnostic) Dataset (UCI Machine Learning Repository / Scikit-Learn).
- **Instance Size:** 569 total instances (426 training instances, 143 test instances via 75/25 stratified split) — *exceeds the minimum requirement of 500 instances*.
- **Feature Size:** 30 continuous numerical input features (mean, standard error, and worst/largest values of 10 cell nucleus properties including radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, and fractal dimension) — *exceeds the minimum requirement of 12 features*.
- **Target Variable:** Binary class `target` (`0` = Malignant: 212 instances, `1` = Benign: 357 instances).
- **Preprocessing Applied:** Z-score standardization (`StandardScaler`) fitted strictly on the training partition and serialized (`scaler_artifact.joblib`) to transform the test set during evaluation.

### c. GitHub Repository Link
- **Repository URL:** `https://github.com/<your-username>/breast-cancer-ml-classifier`
- Contains `app.py`, `train_models.py`, `requirements.txt`, `README.md`, `test_data.csv`, and serialized `.joblib` models inside `model/`.

### d. Models Used & Comparison Table

All five classification models were trained on the same standardized training partition and evaluated on the held-out `test_data.csv` (143 instances, `random_state=725`):

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.9930 | 0.9998 | 1.0000 | 0.9889 | 0.9944 | 0.9852 |
| **Decision Tree** | 0.9231 | 0.9022 | 0.9540 | 0.9222 | 0.9379 | 0.8378 |
| **KNN** | 0.9720 | 0.9856 | 0.9674 | 0.9889 | 0.9780 | 0.9400 |
| **Naive Bayes** | 0.9510 | 0.9945 | 0.9663 | 0.9556 | 0.9609 | 0.8956 |
| **Support Vector Machine** | 0.9860 | 0.9992 | 0.9889 | 0.9889 | 0.9889 | 0.9700 |

### e. Observations on Model Performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Achieved the highest overall accuracy (99.30%), AUC (0.9998), Precision (100.0%), F1 (0.9944), and MCC (0.9852). Because the standardized cytological features exhibit strong linear separability in high-dimensional space, L2-regularized Logistic Regression generalizes exceptionally well without overfitting. |
| **Decision Tree** | Recorded the lowest accuracy (92.31%), AUC (0.9022), and MCC (0.8378). Axis-aligned orthogonal splits struggle to capture smooth diagonal decision boundaries across 30 correlated morphological features, resulting in higher variance on unseen test samples. |
| **KNN** | Delivered strong accuracy (97.20%) and recall (98.89%) using distance-weighted 7-nearest neighbors. While effective after feature standardization, performance is slightly diluted by the curse of dimensionality across all 30 features. |
| **Naive Bayes** | Achieved a high AUC (0.9945) and solid accuracy (95.10%), though its MCC (0.8956) trails the linear and kernel classifiers. Its conditional independence assumption is partially violated because radius, perimeter, area, and concavity features are mathematically colinear. |
| **Support Vector Machine** | Ranked second overall with 98.60% accuracy, 0.9992 AUC, and 0.9700 MCC. The RBF kernel effectively maps non-linear interactions while maintaining wide margin separation between malignant and benign clusters. |
| **Overall, Winner for your dataset?** | **Logistic Regression** is the overall winner for this dataset, leading across every primary metric (Accuracy: 0.9930, AUC: 0.9998, Precision: 1.0000, F1: 0.9944, MCC: 0.9852) with minimal computational overhead. |
