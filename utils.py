"""
utils.py — Helper functions for the Telco Customer Churn Streamlit App.
Handles data loading, model loading, feature engineering, and input preprocessing.
"""

import os
import joblib
import pandas as pd
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

NUMERICAL_COLS = ['tenure', 'MonthlyCharges', 'TotalCharges']

CONTRACT_MAPPING = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}

NOMINAL_COLS = [
    'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling',
    'PaymentMethod'
]

# Exact feature order produced by preprocessing pipeline
FEATURE_NAMES = [
    'SeniorCitizen', 'tenure', 'Contract', 'MonthlyCharges', 'TotalCharges',
    'gender_Male', 'Partner_Yes', 'Dependents_Yes', 'PhoneService_Yes',
    'MultipleLines_No phone service', 'MultipleLines_Yes',
    'InternetService_Fiber optic', 'InternetService_No',
    'OnlineSecurity_No internet service', 'OnlineSecurity_Yes',
    'OnlineBackup_No internet service', 'OnlineBackup_Yes',
    'DeviceProtection_No internet service', 'DeviceProtection_Yes',
    'TechSupport_No internet service', 'TechSupport_Yes',
    'StreamingTV_No internet service', 'StreamingTV_Yes',
    'StreamingMovies_No internet service', 'StreamingMovies_Yes',
    'PaperlessBilling_Yes',
    'PaymentMethod_Credit card (automatic)',
    'PaymentMethod_Electronic check',
    'PaymentMethod_Mailed check'
]


# ──────────────────────────────────────────────────────────────────────────────
# Data Loading
# ──────────────────────────────────────────────────────────────────────────────

def load_data():
    """
    Load the raw Telco Customer Churn CSV and apply the same cleaning
    steps used during preprocessing:
      - Convert TotalCharges to float (coerce errors → NaN)
      - Fill NaN TotalCharges with 0.0
      - Drop customerID column
    Returns a cleaned pandas DataFrame (with original Churn Yes/No intact).
    """
    # Support running from project root OR from notebooks/ subfolder
    candidates = [
        os.path.join('data', 'WA_Fn-UseC_-Telco-Customer-Churn.csv'),
        os.path.join('..', 'data', 'WA_Fn-UseC_-Telco-Customer-Churn.csv'),
    ]
    csv_path = None
    for p in candidates:
        if os.path.exists(p):
            csv_path = p
            break
    if csv_path is None:
        raise FileNotFoundError(
            "Dataset CSV not found. Make sure data/ folder exists relative to app.py."
        )

    df = pd.read_csv(csv_path)

    # Fix TotalCharges type
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(0.0)

    # Drop unique identifier
    df = df.drop(columns=['customerID'])

    return df


# ──────────────────────────────────────────────────────────────────────────────
# Model Loading
# ──────────────────────────────────────────────────────────────────────────────

def load_model(model_name):
    """
    Load a serialized model from models/<model_name>.pkl using joblib.
    Supports running from project root OR from notebooks/ subfolder.
    """
    candidates = [
        os.path.join('models', f'{model_name}.pkl'),
        os.path.join('..', 'models', f'{model_name}.pkl'),
    ]
    for p in candidates:
        if os.path.exists(p):
            return joblib.load(p)
    raise FileNotFoundError(
        f"Model file '{model_name}.pkl' not found in models/ directory."
    )


# ──────────────────────────────────────────────────────────────────────────────
# Feature Names
# ──────────────────────────────────────────────────────────────────────────────

def get_feature_names():
    """
    Return the list of feature names in the exact order used during training.
    Must match the column order of X_train saved in preprocessing.ipynb.
    """
    return FEATURE_NAMES.copy()


# ──────────────────────────────────────────────────────────────────────────────
# Input Preprocessing
# ──────────────────────────────────────────────────────────────────────────────

def preprocess_input(input_dict):
    """
    Accept a dictionary of raw user inputs from the prediction form and
    apply identical encoding as the training pipeline:
      1. Ordinal encode Contract
      2. One-hot encode all nominal columns (drop_first=True)
      3. Align columns to match the exact training feature order
      4. StandardScaler on numerical columns (using the saved scaler)

    Parameters
    ----------
    input_dict : dict
        Keys must include: gender, SeniorCitizen, Partner, Dependents,
        tenure, PhoneService, MultipleLines, InternetService,
        OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport,
        StreamingTV, StreamingMovies, Contract, PaperlessBilling,
        PaymentMethod, MonthlyCharges, TotalCharges

    Returns
    -------
    pd.DataFrame
        A single-row DataFrame ready for model.predict().
    """
    # Build a single-row DataFrame from the input dictionary
    df = pd.DataFrame([input_dict])

    # ── Step 1: Ordinal encode Contract ──
    df['Contract'] = df['Contract'].map(CONTRACT_MAPPING)

    # ── Step 2: One-hot encode nominal columns ──
    cols_to_encode = [c for c in NOMINAL_COLS if c in df.columns]
    df = pd.get_dummies(df, columns=cols_to_encode, drop_first=True, dtype=int)

    # ── Step 3: Align columns — add any missing OHE columns as 0 ──
    for col in FEATURE_NAMES:
        if col not in df.columns:
            df[col] = 0

    # Reorder to match exact training feature order
    df = df[FEATURE_NAMES]

    # Ensure numeric dtype throughout
    df = df.astype(float)

    # ── Step 4: Scale numerical columns using the saved scaler ──
    scaler_candidates = [
        os.path.join('models', 'scaler.pkl'),
        os.path.join('..', 'models', 'scaler.pkl'),
    ]
    scaler_path = None
    for p in scaler_candidates:
        if os.path.exists(p):
            scaler_path = p
            break
    if scaler_path is None:
        raise FileNotFoundError(
            "scaler.pkl not found in models/ directory. "
            "Please run preprocessing.ipynb first."
        )

    scaler = joblib.load(scaler_path)
    df[NUMERICAL_COLS] = scaler.transform(df[NUMERICAL_COLS])

    return df
