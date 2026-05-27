# Customer Churn Prediction — Problem Understanding

This repository contains the machine learning solution for predicting customer churn using the Telco Customer Churn dataset. Below is the detailed Problem Understanding and Business Requirements documentation for the project.

---

## 1. Problem Statement

* **Churn in Telecom**: Customer churn (or attrition) is the phenomenon where subscribers terminate their relationship with a service provider. In the telecommunications industry, this includes active cancellation of phone or internet services, contract non-renewal, or porting a number to a competitor.
* **Prediction Objective**: The task is defined as a supervised **binary classification** problem. The goal is to predict whether a customer will churn in the next billing cycle:
  * **`Churn = 1 (Yes)`**: The customer is predicted to churn.
  * **`Churn = 0 (No)`**: The customer is predicted to remain with the provider.
* **Dataset Source**: The project utilizes the **Telco Customer Churn Dataset (Kaggle)**, containing demographics, services, account information, and churn status for $7,043$ customers.

---

## 2. Business Goal

* **Acquisition vs. Retention Costs**: In a highly saturated telecommunications market, customer acquisition is capital-intensive. It costs **5 to 25 times more** to acquire a new customer (Customer Acquisition Cost - CAC) than it does to retain an existing one (Customer Retention Cost - CRC).
* **Proactive Retention Interventions**: Early churn prediction allows customer-facing teams to intervene proactively before a customer cancels their service. Effective interventions include:
  * **Targeted Loyalty Incentives**: Waived fees, billing credits, or device upgrade discounts for high-value accounts.
  * **Specialized Service Plans**: Re-aligning data limits or features to match the customer's actual usage patterns, reducing costs for under-utilizers.
  * **Contract Re-negotiations**: Offering introductory discount packages to transition high-risk month-to-month subscribers into 1-year or 2-year contracts.
* **Revenue Impact of Undetected Churn**: Undetected churn causes direct erosion of Monthly Recurring Revenue (MRR) and decreases Customer Lifetime Value (CLV). Additionally, it forces the business to spend more on customer acquisition just to maintain a flat user base, severely degrading profitability.

---

## 3. Success Criteria

* **Recall is the Priority**: 
  $$\text{Recall} = \frac{\text{True Positives (TP)}}{\text{True Positives (TP)} + \text{False Negatives (FN)}}$$
  Recall measures the percentage of actual churners the model flags. In this business scenario, missing a churner (False Negative) means the customer is lost forever, whereas flagging a customer who wasn't going to leave (False Positive) only results in a minor cost for a retention discount. Therefore, maximizing Recall is the primary objective.
* **Accuracy is Misleading**: The dataset exhibits class imbalance (approximately $73.5\%$ active, $26.5\%$ churned). A naive classifier predicting "No Churn" for every customer would achieve $73.5\%$ accuracy but fail to identify a single at-risk customer. Accuracy is therefore an insufficient metric.
* **F1 Score**: 
  $$\text{F1 Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$
  The F1 Score provides a balanced harmonic mean of Precision and Recall. It ensures that while we maximize Recall, we do not completely collapse Precision (which would lead to over-predicting churn and wasting retention budgets on stable customers).

---

## 4. Stakeholders

* **Marketing Teams**: Use risk segmentations to design targeted, high-ROI retention campaigns and determine the optimal allocation of promotional budgets.
* **Customer Success Managers**: Prioritize accounts flagged as high-risk, using tailored conversation scripts and billing concessions during support calls.
* **Product Teams**: Identify specific services (e.g., fiber optic lines or lack of tech support) that correlate with high churn rates, informing feature updates and service improvements.
* **Finance Team**: Predict revenue retention, forecast future cash flows (MRR), and calculate the financial returns of retention campaigns compared to acquisition spend.

---

## 5. Proposed Solution

* **Machine Learning Pipeline**: A comprehensive classification pipeline will be built to preprocess features, handle class imbalance, and evaluate model performance. We will compare performance across **6 algorithms**:
  1. Logistic Regression
  2. Decision Tree
  3. Random Forest
  4. K-Nearest Neighbors (KNN)
  5. Support Vector Machine (SVM)
  6. XGBoost (Extreme Gradient Boosting)
* **Interactive Deployment**: The best-performing model (optimized for Recall and F1 Score) will be exported and deployed as an interactive **Streamlit dashboard**. This dashboard will allow stakeholders to input customer attributes, obtain real-time churn probabilities, and view recommended retention strategies.
