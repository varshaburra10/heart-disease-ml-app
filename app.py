import streamlit as st, pandas as pd, joblib
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef, confusion_matrix,
    ConfusionMatrixDisplay, classification_report)

TARGET = "target"
MODEL_FILES = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "KNN": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "SVM": "model/svm.joblib",
}

st.set_page_config(page_title="Heart Disease Classifier Lab", layout="wide")
st.title("Heart Disease Classifier Lab")
st.caption("Compare five trained classifiers on your own test CSV. "
           "Models are pre-trained and are not retrained on upload.")

with st.sidebar:
    st.header("Controls")
    file = st.file_uploader("Upload test CSV (features + 'target')", type="csv")
    choice = st.selectbox("Select model", list(MODEL_FILES))

if file is None:
    st.info("Upload test_data.csv from the sidebar to begin.")
    st.stop()

df = pd.read_csv(file)
if TARGET not in df.columns:
    st.error(f"Column '{TARGET}' is missing from the uploaded file.")
    st.stop()

X, y = df.drop(columns=[TARGET]), df[TARGET]
model = joblib.load(MODEL_FILES[choice])
try:
    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
except Exception as e:
    st.error(f"Could not score this file. Check the columns match the training data.\n\n{e}")
    st.stop()

st.subheader(f"Results: {choice}  ({len(df)} rows)")
c = st.columns(6)
c[0].metric("Accuracy", f"{accuracy_score(y, pred):.3f}")
c[1].metric("AUC", f"{roc_auc_score(y, proba):.3f}")
c[2].metric("Precision", f"{precision_score(y, pred):.3f}")
c[3].metric("Recall", f"{recall_score(y, pred):.3f}")
c[4].metric("F1", f"{f1_score(y, pred):.3f}")
c[5].metric("MCC", f"{matthews_corrcoef(y, pred):.3f}")

left, right = st.columns(2)
with left:
    st.markdown("**Confusion matrix**")
    fig, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay(confusion_matrix(y, pred),
                           display_labels=["No disease", "Disease"]).plot(ax=ax, cmap="Reds")
    st.pyplot(fig)
with right:
    st.markdown("**Classification report**")
    st.code(classification_report(y, pred))