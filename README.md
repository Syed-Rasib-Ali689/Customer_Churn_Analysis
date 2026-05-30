# 📞 Customer Churn Prediction Dashboard

> **Live App:** `https://customerchurnanalysisbysyedrasibali.streamlit.app/` ← Replace after deployment

A complete end-to-end Machine Learning pipeline for predicting Telco customer churn. Six classification models are trained, benchmarked, and served through an interactive Streamlit dashboard with real-time prediction capability.

---

## 📌 Business Problem

Customer churn costs telecom providers **5–25× more** than retaining existing customers. This project builds a binary classifier to identify high-risk subscribers **before** they cancel, enabling marketing and customer success teams to launch targeted retention interventions — reducing Monthly Recurring Revenue (MRR) erosion and improving Customer Lifetime Value (CLV).

---

## 📊 Dataset

| Property | Value |
|---|---|
| Source | [Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) |
| Records | 7,043 customers |
| Features | 20 (demographics, services, account info) |
| Target | `Churn` — Yes (26.54%) / No (73.46%) |
| Class Imbalance | Handled with SMOTE on training set only |

---

## 🛠 Tech Stack

| Library | Version | Purpose |
|---|---|---|
| Python | 3.10+ | Core language |
| Streamlit | ≥1.35 | Interactive dashboard |
| Scikit-learn | ≥1.4 | ML models, preprocessing, metrics |
| XGBoost | ≥2.0 | Gradient boosting classifier |
| Imbalanced-learn | ≥0.12 | SMOTE oversampling |
| Plotly | ≥5.20 | Interactive visualizations |
| Pandas | ≥2.0 | Data manipulation |
| Joblib | ≥1.3 | Model serialization |

---

## 📁 Project Structure

```
project/
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Raw dataset
├── notebooks/
│   ├── dataset_analysis.ipynb                  # Phase 2 — Dataset understanding
│   ├── eda_analysis.ipynb                      # Phase 3 — Exploratory data analysis
│   ├── preprocessing.ipynb                     # Phase 4 — Data preprocessing pipeline
│   ├── model_training.ipynb                    # Phase 5 — Train & evaluate 6 models
│   ├── model_comparison.ipynb                  # Phase 6 — Comparison charts & champion selection
│   └── hyperparameter_tuning.ipynb             # Phase 7 — GridSearchCV & RandomizedSearchCV
├── models/
│   ├── scaler.pkl                              # Fitted StandardScaler
│   ├── logistic_regression.pkl                 # Trained model
│   ├── decision_tree.pkl                       # Trained model
│   ├── random_forest.pkl                       # Trained model
│   ├── knn.pkl                                 # Trained model
│   ├── svm.pkl                                 # Trained model
│   ├── xgboost.pkl                             # Trained model
│   ├── random_forest_tuned.pkl                 # Tuned RF model
│   ├── xgboost_tuned.pkl                       # Tuned XGBoost model
│   ├── results_summary.csv                     # All model metrics
│   └── tuning_comparison.csv                   # Baseline vs tuned comparison
├── processed/
│   ├── X_train.pkl                             # SMOTE-resampled training features
│   ├── X_test.pkl                              # Test features
│   ├── y_train.pkl                             # Training labels
│   └── y_test.pkl                              # Test labels
├── visuals/
│   └── churn_distribution_chart.png            # EDA export
├── app.py                                      # Streamlit application entry point
├── utils.py                                    # Helper functions (load, preprocess, predict)
├── requirements.txt                            # Pinned dependencies
└── README.md                                   # This file
```

---

## 🔍 Key EDA Findings

- **Month-to-month contracts** have the highest churn rate (~43%), compared to ~11% for two-year contracts — contract type is the strongest churn predictor.
- **Fiber optic internet customers** churn at nearly double the rate of DSL users, suggesting service quality or pricing issues.
- **Customers with short tenure (0–12 months)** are significantly more likely to churn; loyalty increases sharply after 24 months.
- **Electronic check payment** users churn at ~45% vs ~15–18% for automatic payment methods, indicating billing friction is a risk factor.
- **High monthly charges (>$70/month)** correlate strongly with churn, particularly in the absence of tech support or online security services.

---

## 🏆 Model Evaluation Leaderboard

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression** ⭐ | 0.7608 | 0.5354 | 0.7487 | **0.6243** | **0.8352** |
| SVM | 0.7637 | 0.5412 | 0.7193 | 0.6177 | 0.8220 |
| XGBoost | 0.7622 | 0.5419 | 0.6738 | 0.6007 | 0.8089 |
| Random Forest | 0.7651 | 0.5504 | 0.6283 | 0.5868 | 0.8192 |
| KNN | 0.7204 | 0.4826 | 0.7433 | 0.5853 | 0.7821 |
| Decision Tree | 0.7062 | 0.4565 | 0.5615 | 0.5036 | 0.6598 |

**Champion: Logistic Regression** — Highest F1 (0.6243) and ROC-AUC (0.8352). It also achieves the highest Recall (0.7487), meaning it catches the most actual churners — the primary business objective.

---

## 🚀 Local Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the Streamlit app
streamlit run app.py
```

> **Note:** The `processed/` and `models/` files are included. If you want to re-run the full pipeline, execute the notebooks in order: `preprocessing.ipynb` → `model_training.ipynb` → `hyperparameter_tuning.ipynb` → `model_comparison.ipynb`

---

## ☁ Deployment (Streamlit Community Cloud)

1. Push this repository to a **public GitHub repo**
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Connect your GitHub repo
4. Set **Main file path**: `app.py`
5. Click **Deploy**
6. Copy the live URL and update the badge at the top of this README

---

## 📸 Screenshots

> Add 3–4 dashboard screenshots here after deployment (Home, EDA, Model Comparison, Prediction pages).
