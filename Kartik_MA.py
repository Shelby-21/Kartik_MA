# -*- coding: utf-8 -*-
"""loan_default.py - Final Code with Theme 5 (Cool Tones)"""

import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score 
import numpy as np

# --- 1. SETUP & DATA LOADING ---
st.set_page_config(page_title="Loan Default Marketing Analytics", layout="wide")

# Function to load data
@st.cache_data
def load_data():
    try:
        # NOTE: Using 'loan_default.csv' for robust Streamlit Cloud deployment
        # Ensure 'loan_default.csv' is committed to your GitHub repository!
        df = pd.read_csv('loan_default.csv') 
        return df
    except FileNotFoundError:
        st.error("File 'loan_default.csv' not found. Please ensure the CSV data file is in the repository.")
        return None

df = load_data()

if df is not None:
    # --- 2. DATA PREPROCESSING & MODELING (Risk Scoring) ---
    
    df_model = df.copy()
    le_dict = {}
    for col in df_model.columns:
        le = LabelEncoder()
        # Handle potential NaNs in categorical columns by converting to string
        df_model[col] = le.fit_transform(df_model[col].astype(str))
        le_dict[col] = le

    # Define X (Features) and y (Target)
    X = df_model.drop('Default', axis=1)
    y = df_model['Default']

    # Train a simple Decision Tree (max_depth=5)
    clf = DecisionTreeClassifier(max_depth=5, random_state=42)
    clf.fit(X, y)

    # Get probabilities for the 'Yes' class (Risk Score)
    yes_index = list(le_dict['Default'].classes_).index('Yes')
    df['Risk_Probability'] = clf.predict_proba(X)[:, yes_index]

    # --- Confusion Matrix and Metrics Calculation ---
    y_pred = clf.predict(X)
    cm = confusion_matrix(y, y_pred)
    
    TN = cm[0, 0] 
    FP = cm[0, 1] 
    FN = cm[1, 0] 
    TP = cm[1, 1] 
    accuracy = accuracy_score(y, y_pred)
    
    total_actual_no = TN + FP
    total_actual_yes = FN + TP

    type_i_error_rate = FP / total_actual_no if total_actual_no > 0 else 0
    type_ii_error_rate = FN / total_actual_yes if total_actual_yes > 0 else 0


    # --- 3. DASHBOARD LAYOUT ---

    st.title("Loan Default Marketing Analytics Dashboard")
    st.markdown("Analysis of customer profiles and model performance to identify high-risk segments.")
    st.divider()

    # --- TOP ROW: KPI (Updated with Model Metrics) ---
    total_defaults = df[df['Default'] == 'Yes'].shape[0]
    total_customers = df.shape[0]
    default_rate = (total_defaults / total_customers) * 100

    col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
    with col_kpi1:
        st.metric(label="Overall Default Rate", value=f"{default_rate:.1f}%", delta_color="inverse")
    with col_kpi2:
        st.metric(label="Model Accuracy", value=f"{accuracy*100:.1f}%")
    with col_kpi3:
        st.metric(label="False Pos. Rate (Type I Error)", value=f"{type_i_error_rate*100:.1f}%")
    with col_kpi4:
        st.metric(label="False Neg. Rate (Type II Error)", value=f"{type_ii_error_rate*100:.1f}%")

    st.divider()

    # Helper function to calculate risk rate by category
    def get_risk_by_category(column_name):
        risk_df = df.groupby(column_name)['Default'].apply(lambda x: (x == 'Yes').sum() / len(x) * 100).reset_index()
        risk_df.columns = [column_name, 'Default Rate (%)']
        return risk_df.sort_values('Default Rate (%)', ascending=False)


    # --- MIDDLE ROW: THE DRIVERS (Bar Charts - Theme 5 Applied) ---
    st.subheader("The Drivers: Risk Analysis")

    row2_col1, row2_col2 = st.columns(2)

    # Chart 1: Risk by Employment
    with row2_col1:
        emp_risk = get_risk_by_category('Employment_Type')
        fig_emp = px.bar(emp_risk, x='Employment_Type', y='Default Rate (%)',
                         title="Risk by Employment Type",
                         color='Default Rate (%)', 
                         color_continuous_scale='Electric') # <--- THEME 5: ELECTRIC
        st.plotly_chart(fig_emp, use_container_width=True)

    # Chart 2: Risk by Credit History
    with row2_col2:
        cred_risk = get_risk_by_category('Credit_History')
        fig_cred = px.bar(cred_risk, x='