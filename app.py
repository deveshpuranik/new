import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.metrics import roc_curve, roc_auc_score

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(page_title="Fraud Detection System", layout="wide")
st.title("💳 Credit Card Fraud Detection System")
st.markdown("Production-style ML system using a pre-trained model")

# --------------------------------------------------
# LOAD DATA (EDA ONLY)
# --------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("creditcard.csv")

df = load_data()

# --------------------------------------------------
# LOAD TRAINED MODEL (.PKL)
# --------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("fraud_detection_model.pkl")

model = load_model()

# --------------------------------------------------
# EDA SECTION
# --------------------------------------------------
st.header("📊 Exploratory Data Analysis")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Fraud vs Legit Transactions")
    class_counts = df["Class"].value_counts()
    fig, ax = plt.subplots()
    ax.bar(["Legit", "Fraud"], class_counts)
    ax.set_ylabel("Count")
    st.pyplot(fig)

with col2:
    st.subheader("Transaction Amount Distribution")
    fig2, ax2 = plt.subplots()
    ax2.hist(df[df["Class"] == 0]["Amount"], bins=50, alpha=0.6, label="Legit")
    ax2.hist(df[df["Class"] == 1]["Amount"], bins=50, alpha=0.6, label="Fraud")
    ax2.legend()
    ax2.set_xlabel("Transaction Amount")
    st.pyplot(fig2)

# --------------------------------------------------
# MODEL EVALUATION (FAST + CORRECT)
# --------------------------------------------------
st.header("🤖 Model Evaluation (Sampled for Performance)")

X = df.drop("Class", axis=1)
y = df["Class"]

@st.cache_resource
def evaluate_model(_model, X, y):
    X_sample = X.sample(10000, random_state=42)
    y_sample = y.loc[X_sample.index]

    y_prob = _model.predict_proba(X_sample)[:, 1]
    roc_auc = roc_auc_score(y_sample, y_prob)
    fpr, tpr, _ = roc_curve(y_sample, y_prob)

    return roc_auc, fpr, tpr

roc_auc, fpr, tpr = evaluate_model(model, X, y)

st.metric("ROC-AUC Score (Sampled)", round(roc_auc, 3))

fig3, ax3 = plt.subplots()
ax3.plot(fpr, tpr, label="Logistic Regression")
ax3.plot([0, 1], [0, 1], linestyle="--")
ax3.set_xlabel("False Positive Rate")
ax3.set_ylabel("True Positive Rate")
ax3.legend()
st.pyplot(fig3)

# --------------------------------------------------
# REAL-TIME FRAUD PREDICTION
# --------------------------------------------------
st.header("🔍 Real-Time Fraud Prediction")

st.markdown("Enter transaction details to assess fraud risk")

amount = st.number_input("Transaction Amount", 0.0, 50000.0, step=10.0)
time = st.number_input("Transaction Time (seconds)", 0.0)

# Input vector (same structure as training data)
input_features = np.zeros(30)
input_features[0] = time
input_features[-1] = amount

if st.button("Check Fraud Risk"):
    prob = model.predict_proba([input_features])[0][1]

    if prob < 0.3:
        risk = "LOW RISK"
        color = "green"
    elif prob < 0.7:
        risk = "MEDIUM RISK"
        color = "orange"
    else:
        risk = "HIGH RISK"
        color = "red"

    st.subheader("Prediction Result")
    st.metric("Fraud Probability", f"{prob:.2%}")
    st.markdown(f"### Risk Level: **:{color}[{risk}]**")

# --------------------------------------------------
# BUSINESS INSIGHT
# --------------------------------------------------
st.header("📌 Business Insight")

st.markdown("""
- Fraud detection is a **highly imbalanced problem**
- Accuracy is misleading → **ROC-AUC & Recall matter**
- The model outputs **probability scores**
- Thresholds can be adjusted based on **business cost**
""")
