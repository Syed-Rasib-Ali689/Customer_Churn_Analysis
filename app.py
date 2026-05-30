"""
app.py — Telco Customer Churn Prediction Streamlit Application
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    roc_curve, precision_recall_curve, average_precision_score
)
from sklearn.feature_selection import mutual_info_classif
from utils import load_data, load_model, preprocess_input, get_feature_names

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Telco Churn Predictor",
    page_icon="📊",
    layout="wide"
)

# ── Cached Loaders ────────────────────────────────────────────────────────────
@st.cache_data
def cached_load_data():
    return load_data()

@st.cache_resource
def cached_load_model(name):
    return load_model(name)

@st.cache_data
def load_results():
    path = os.path.join('models', 'results_summary.csv')
    df = pd.read_csv(path)
    # Prettify model names for display
    name_map = {
        'logistic_regression': 'Logistic Regression',
        'decision_tree': 'Decision Tree',
        'random_forest': 'Random Forest',
        'knn': 'KNN',
        'svm': 'SVM',
        'xgboost': 'XGBoost'
    }
    df['Model'] = df['Model'].map(name_map).fillna(df['Model'])
    return df

@st.cache_data
def load_processed():
    d = 'processed'
    return (
        joblib.load(os.path.join(d, 'X_train.pkl')),
        joblib.load(os.path.join(d, 'X_test.pkl')),
        joblib.load(os.path.join(d, 'y_train.pkl')),
        joblib.load(os.path.join(d, 'y_test.pkl')),
    )

# ── Sidebar Navigation ───────────────────────────────────────────────────────
page = st.sidebar.radio(
    "📌 Navigation",
    ["🏠 Home", "📋 Dataset Explorer", "📊 EDA Dashboard",
     "🤖 Model Training", "🏆 Model Comparison", "🔮 Prediction"]
)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Home
# ═════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.title("📞 Telco Customer Churn Prediction")

    st.markdown("""
    **Customer churn** is the phenomenon where subscribers terminate their
    relationship with a service provider. In the highly competitive
    telecommunications industry, it costs **5–25× more** to acquire a new
    customer than to retain an existing one. Early churn prediction allows
    teams to intervene proactively and preserve revenue.

    This application provides an **end-to-end machine-learning pipeline**:
    from raw data exploration, through model training and comparison, to
    real-time churn predictions. Six classification algorithms have been
    trained and benchmarked so stakeholders can pick the best strategy.

    The goal is to maximise **Recall** (catching every at-risk customer)
    while maintaining a healthy **F1 Score** (avoiding excessive false
    alarms that waste retention budgets).
    """)

    st.divider()
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Records", "7,043")
    c2.metric("Features", "20")
    c3.metric("Churn Rate", "26.54 %")

    st.divider()
    st.info("🛠 **Tech Stack:** Python · Streamlit · Scikit-learn · XGBoost · Plotly · Pandas · Joblib")

    st.subheader("🗺 Navigation Guide")
    guide = {
        "📋 Dataset Explorer": "Browse and search the raw dataset, view statistics, and plot value counts.",
        "📊 EDA Dashboard": "Interactive univariate, bivariate, correlation, and feature-importance charts.",
        "🤖 Model Training": "Pick any of the 6 models, evaluate it, and inspect the confusion matrix.",
        "🏆 Model Comparison": "Leaderboard, ROC curves, PR curves, and feature importance comparison.",
        "🔮 Prediction": "Enter a customer's details and get a real-time churn risk assessment.",
    }
    for k, v in guide.items():
        st.markdown(f"- **{k}** — {v}")

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Dataset Explorer
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📋 Dataset Explorer":
    st.title("📋 Dataset Explorer")
    df = cached_load_data()

    st.subheader("Full Dataset")
    st.dataframe(df, use_container_width=True)

    st.subheader("Descriptive Statistics")
    st.write(df.describe())

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Null Values", int(df.isnull().sum().sum()))

    st.subheader("Data Types & Null Counts")
    info_df = pd.DataFrame({
        "Column": df.columns,
        "Dtype": [str(d) for d in df.dtypes],
        "Non-Null Count": df.notnull().sum().values,
        "Null Count": df.isnull().sum().values
    })
    st.dataframe(info_df, use_container_width=True)

    st.subheader("Column Value Counts")
    selected_col = st.selectbox("Select a column", df.columns.tolist())
    vc = df[selected_col].value_counts().reset_index()
    vc.columns = [selected_col, 'Count']
    fig = px.bar(vc, x=selected_col, y='Count',
                 title=f"Value Counts — {selected_col}",
                 color='Count', color_continuous_scale='Blues')
    st.plotly_chart(fig, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3 — EDA Dashboard
# ═════════════════════════════════════════════════════════════════════════════
elif page == "📊 EDA Dashboard":
    st.title("📊 Exploratory Data Analysis")
    df = cached_load_data()

    chart_type = st.selectbox("Select Chart Category", [
        "A: Univariate Analysis",
        "B: Bivariate Analysis",
        "C: Correlation Heatmap",
        "D: Mutual Information Feature Importance"
    ])

    if chart_type.startswith("A"):
        feat = st.selectbox("Feature", ["SeniorCitizen", "MonthlyCharges", "tenure"])
        if feat == "SeniorCitizen":
            vc = df[feat].value_counts().reset_index()
            vc.columns = [feat, 'Count']
            vc[feat] = vc[feat].map({0: 'Non-Senior (0)', 1: 'Senior (1)'})
            fig = px.bar(vc, x=feat, y='Count', color=feat,
                         title="SeniorCitizen Distribution",
                         color_discrete_sequence=['#2A9D8F', '#E76F51'])
        elif feat == "MonthlyCharges":
            fig = px.histogram(df, x='MonthlyCharges', nbins=40, marginal='box',
                               title="MonthlyCharges Distribution",
                               color_discrete_sequence=['#636EFA'])
        else:
            fig = px.histogram(df, x='tenure', nbins=40, marginal='box',
                               title="Tenure Distribution",
                               color_discrete_sequence=['#E76F51'])
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type.startswith("B"):
        feat = st.selectbox("Feature", [
            "Contract", "PaymentMethod", "InternetService", "tenure", "MonthlyCharges"
        ])
        if feat in ["Contract", "PaymentMethod", "InternetService"]:
            ct = df.groupby([feat, 'Churn']).size().reset_index(name='Count')
            fig = px.bar(ct, x=feat, y='Count', color='Churn', barmode='group',
                         title=f"Churn by {feat}",
                         color_discrete_map={'Yes': '#E76F51', 'No': '#2A9D8F'})
        elif feat == "tenure":
            fig = px.box(df, x='Churn', y='tenure', color='Churn',
                         title="Tenure Distribution by Churn",
                         color_discrete_map={'Yes': '#E76F51', 'No': '#2A9D8F'})
        else:
            fig = px.violin(df, x='Churn', y='MonthlyCharges', color='Churn',
                            box=True, title="MonthlyCharges Distribution by Churn",
                            color_discrete_map={'Yes': '#E76F51', 'No': '#2A9D8F'})
        st.plotly_chart(fig, use_container_width=True)

    elif chart_type.startswith("C"):
        df_num = df.copy()
        df_num['Churn_num'] = df_num['Churn'].map({'Yes': 1, 'No': 0})
        num_df = df_num[['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen', 'Churn_num']]
        corr = num_df.corr()
        fig = px.imshow(corr, text_auto=".2f", color_continuous_scale='RdBu_r',
                        title="Correlation Heatmap — Numerical Features", aspect='auto')
        st.plotly_chart(fig, use_container_width=True)

    else:  # D — Mutual Information
        df_enc = df.copy()
        df_enc['Churn'] = df_enc['Churn'].map({'Yes': 1, 'No': 0})
        df_enc['Contract'] = df_enc['Contract'].map(
            {'Month-to-month': 0, 'One year': 1, 'Two year': 2})
        nom = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
               'InternetService', 'OnlineSecurity', 'OnlineBackup',
               'DeviceProtection', 'TechSupport', 'StreamingTV',
               'StreamingMovies', 'PaperlessBilling', 'PaymentMethod']
        df_enc = pd.get_dummies(df_enc, columns=nom, drop_first=True, dtype=int)
        X_mi = df_enc.drop(columns=['Churn'])
        y_mi = df_enc['Churn']
        mi = mutual_info_classif(X_mi, y_mi, random_state=42)
        mi_df = pd.DataFrame({'Feature': X_mi.columns, 'MI Score': mi})
        mi_df = mi_df.sort_values('MI Score', ascending=True).tail(15)
        fig = px.bar(mi_df, x='MI Score', y='Feature', orientation='h',
                     title="Top 15 Features — Mutual Information",
                     color='MI Score', color_continuous_scale='Viridis')
        st.plotly_chart(fig, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Model Training
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🤖 Model Training":
    st.title("🤖 Model Training & Evaluation")

    model_map = {
        'Logistic Regression': 'logistic_regression',
        'Decision Tree': 'decision_tree',
        'Random Forest': 'random_forest',
        'K-Nearest Neighbors': 'knn',
        'Support Vector Machine': 'svm',
        'XGBoost': 'xgboost'
    }
    choice = st.selectbox("Select Model", list(model_map.keys()))

    if st.button("▶ Evaluate Model"):
        with st.spinner("Loading model and computing metrics…"):
            _, X_test, _, y_test = load_processed()
            model = cached_load_model(model_map[choice])

            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1]

            acc  = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec  = recall_score(y_test, y_pred)
            f1   = f1_score(y_test, y_pred)
            auc  = roc_auc_score(y_test, y_prob)
            cm   = confusion_matrix(y_test, y_pred)

        st.subheader(f"Results — {choice}")
        c1, c2, c3 = st.columns(3)
        c1.metric("Accuracy", f"{acc:.4f}")
        c2.metric("Precision", f"{prec:.4f}")
        c3.metric("Recall", f"{rec:.4f}")

        c4, c5, _ = st.columns(3)
        c4.metric("F1 Score", f"{f1:.4f}")
        c5.metric("ROC-AUC", f"{auc:.4f}")

        st.subheader("Confusion Matrix")
        fig = px.imshow(cm, text_auto=True, color_continuous_scale='Blues',
                        x=['No Churn', 'Churn'], y=['No Churn', 'Churn'],
                        labels=dict(x="Predicted", y="Actual"),
                        title=f"{choice} — Confusion Matrix")
        st.plotly_chart(fig, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 5 — Model Comparison
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🏆 Model Comparison":
    st.title("🏆 Model Comparison Dashboard")

    results_df = load_results()
    champion = results_df.iloc[0]
    st.success(f"🏅 Champion: **{champion['Model']}** — F1: {champion['F1']:.4f} | ROC-AUC: {champion['ROC-AUC']:.4f}")

    st.subheader("Metrics Leaderboard")

    def highlight_champion(row):
        if row['Model'] == champion['Model']:
            return ['background-color: #d4edda; color: #155724'] * len(row)
        return [''] * len(row)

    styled = results_df.style.apply(highlight_champion, axis=1).format(
        {'Accuracy': '{:.4f}', 'Precision': '{:.4f}', 'Recall': '{:.4f}',
         'F1': '{:.4f}', 'ROC-AUC': '{:.4f}'})
    st.dataframe(styled, use_container_width=True)

    chart_opt = st.selectbox("Select Comparison Chart", [
        "Accuracy vs F1 Score",
        "ROC Curves",
        "Confusion Matrices",
        "Feature Importance (RF vs XGBoost)",
        "Precision-Recall Curves"
    ])

    model_files = {
        'Logistic Regression': 'logistic_regression',
        'Decision Tree': 'decision_tree',
        'Random Forest': 'random_forest',
        'KNN': 'knn',
        'SVM': 'svm',
        'XGBoost': 'xgboost'
    }

    if chart_opt == "Accuracy vs F1 Score":
        fig = go.Figure(data=[
            go.Bar(name='Accuracy', x=results_df['Model'], y=results_df['Accuracy'],
                   text=results_df['Accuracy'].round(4), textposition='auto',
                   marker_color='rgb(31,119,180)'),
            go.Bar(name='F1 Score', x=results_df['Model'], y=results_df['F1'],
                   text=results_df['F1'].round(4), textposition='auto',
                   marker_color='coral')
        ])
        fig.update_layout(barmode='group', title="Accuracy vs F1 Score — All Models",
                          template='plotly_white', yaxis_range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)

    elif chart_opt == "ROC Curves":
        _, X_test, _, y_test = load_processed()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                      line=dict(dash='dash', color='gray'), name='Random (AUC=0.50)'))
        for name, fname in model_files.items():
            mdl = cached_load_model(fname)
            yp = mdl.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, yp)
            a = roc_auc_score(y_test, yp)
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',
                          name=f"{name} (AUC={a:.2f})"))
        fig.update_layout(title="ROC Curves — All Models",
                          xaxis_title="False Positive Rate",
                          yaxis_title="True Positive Rate",
                          template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)

    elif chart_opt == "Confusion Matrices":
        _, X_test, _, y_test = load_processed()
        names = list(model_files.keys())
        fig = make_subplots(rows=2, cols=3, subplot_titles=names,
                            horizontal_spacing=0.1, vertical_spacing=0.18)
        for idx, (name, fname) in enumerate(model_files.items()):
            mdl = cached_load_model(fname)
            cm = confusion_matrix(y_test, mdl.predict(X_test))
            r, c = divmod(idx, 3)
            fig.add_trace(go.Heatmap(
                z=cm, x=['No Churn', 'Churn'], y=['No Churn', 'Churn'],
                colorscale='Blues', showscale=False,
                text=[[str(v) for v in row] for row in cm],
                texttemplate="%{text}", textfont={"size": 14}),
                row=r + 1, col=c + 1)
        fig.update_yaxes(autorange="reversed")
        fig.update_layout(title_text="Confusion Matrices — All Models",
                          height=600, template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)

    elif chart_opt == "Feature Importance (RF vs XGBoost)":
        rf  = cached_load_model('random_forest')
        xgb = cached_load_model('xgboost')
        feats = get_feature_names()
        imp_df = pd.DataFrame({
            'Feature': feats,
            'Random Forest': rf.feature_importances_,
            'XGBoost': xgb.feature_importances_
        })
        imp_df['Avg'] = (imp_df['Random Forest'] + imp_df['XGBoost']) / 2
        top = imp_df.nlargest(15, 'Avg').iloc[::-1]
        fig = go.Figure(data=[
            go.Bar(name='Random Forest', y=top['Feature'], x=top['Random Forest'],
                   orientation='h', marker_color='#3B6D11'),
            go.Bar(name='XGBoost', y=top['Feature'], x=top['XGBoost'],
                   orientation='h', marker_color='#FFBF00')
        ])
        fig.update_layout(barmode='group',
            title="Feature Importance — Random Forest vs XGBoost",
            xaxis_title="Importance Score", yaxis_title="Feature",
            template='plotly_white', height=700)
        st.plotly_chart(fig, use_container_width=True)

    else:  # PR Curves
        _, X_test, _, y_test = load_processed()
        fig = go.Figure()
        for name, fname in model_files.items():
            mdl = cached_load_model(fname)
            yp = mdl.predict_proba(X_test)[:, 1]
            prec_arr, rec_arr, _ = precision_recall_curve(y_test, yp)
            ap = average_precision_score(y_test, yp)
            fig.add_trace(go.Scatter(x=rec_arr, y=prec_arr, mode='lines',
                          name=f"{name} (AP={ap:.2f})"))
        fig.update_layout(title="Precision-Recall Curves — All Models",
                          xaxis_title="Recall", yaxis_title="Precision",
                          template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# PAGE 6 — Prediction Form
# ═════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Prediction":
    st.title("🔮 Customer Churn Risk Predictor")
    st.markdown("Fill in the customer details below and click **Predict**.")

    c1, c2, c3 = st.columns(3)

    with c1:
        gender     = st.selectbox("Gender", ["Male", "Female"])
        senior     = st.checkbox("Senior Citizen")
        partner    = st.checkbox("Partner")
        dependents = st.checkbox("Dependents")
        tenure     = st.slider("Tenure (months)", 0, 72, 12)

    with c2:
        phone      = st.selectbox("Phone Service", ["Yes", "No"])
        multi      = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        internet   = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        security   = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        backup     = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])

    with c3:
        protection  = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        techsupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        tv          = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        movies      = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
        contract    = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

    c4, c5 = st.columns(2)
    with c4:
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment   = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ])
    with c5:
        monthly = st.slider("Monthly Charges ($)", 0.0, 120.0, 50.0, step=0.5)
        total   = round(monthly * tenure, 2)
        st.info(f"Estimated Total Charges: **${total:.2f}**")

    if st.button("🔮 Predict Churn Risk", type="primary"):
        input_dict = {
            'gender':          gender,
            'SeniorCitizen':   int(senior),
            'Partner':         'Yes' if partner    else 'No',
            'Dependents':      'Yes' if dependents else 'No',
            'tenure':          tenure,
            'PhoneService':    phone,
            'MultipleLines':   multi,
            'InternetService': internet,
            'OnlineSecurity':  security,
            'OnlineBackup':    backup,
            'DeviceProtection': protection,
            'TechSupport':     techsupport,
            'StreamingTV':     tv,
            'StreamingMovies': movies,
            'Contract':        contract,
            'PaperlessBilling': paperless,
            'PaymentMethod':   payment,
            'MonthlyCharges':  monthly,
            'TotalCharges':    total,
        }

        try:
            processed = preprocess_input(input_dict)
        except Exception as e:
            st.error(f"Preprocessing error: {e}")
            st.stop()

        # Try best models first, fall back gracefully
        model_used = None
        for m in ['xgboost_tuned', 'random_forest_tuned', 'xgboost', 'random_forest']:
            try:
                model = cached_load_model(m)
                model_used = m
                break
            except (FileNotFoundError, Exception):
                continue

        if model_used is None:
            st.error("No model file found. Please run the training notebooks first.")
            st.stop()

        pred = model.predict(processed)[0]
        prob = model.predict_proba(processed)[0]

        st.divider()
        st.caption(f"Model used: `{model_used}`")

        if pred == 1:
            st.error("⚠ **High Churn Risk**")
            st.metric("Churn Probability", f"{prob[1] * 100:.1f} %")
            st.warning("💡 **Action 1:** Offer a discounted upgrade to a 1-year or 2-year contract.")
            st.warning("💡 **Action 2:** Provide a loyalty credit or billing adjustment to reduce monthly cost pressure.")
            st.warning("💡 **Action 3:** Assign a dedicated account manager for personalised outreach and service review.")
        else:
            st.success("✅ **Low Churn Risk**")
            st.metric("Retention Confidence", f"{prob[0] * 100:.1f} %")
            st.info("This customer is stable. Continue standard engagement and monitor for future changes.")
