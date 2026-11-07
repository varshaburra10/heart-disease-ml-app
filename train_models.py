import pandas as pd, joblib, os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef)

TARGET = "target"
NUM = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CAT = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]

df = pd.read_csv("heart.csv")
X, y = df.drop(columns=[TARGET]), df[TARGET]

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=2026, stratify=y)

pd.concat([X_te, y_te], axis=1).to_csv("test_data.csv", index=False)

def prep():
    return ColumnTransformer([
        ("num", StandardScaler(), NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT),
    ])

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=2026),
    "KNN": KNeighborsClassifier(n_neighbors=7),
    "Naive Bayes": GaussianNB(),
    "SVM": SVC(kernel="rbf", probability=True, random_state=2026),
}

os.makedirs("model", exist_ok=True)
rows = []
for name, clf in models.items():
    pipe = Pipeline([("prep", prep()), ("clf", clf)])
    pipe.fit(X_tr, y_tr)
    pred = pipe.predict(X_te)
    proba = pipe.predict_proba(X_te)[:, 1]
    rows.append({
        "ML Model Name": name,
        "Accuracy": accuracy_score(y_te, pred),
        "AUC": roc_auc_score(y_te, proba),
        "Precision": precision_score(y_te, pred),
        "Recall": recall_score(y_te, pred),
        "F1": f1_score(y_te, pred),
        "MCC": matthews_corrcoef(y_te, pred),
    })
    joblib.dump(pipe, "model/" + name.replace(" ", "_").lower() + ".joblib")

res = pd.DataFrame(rows).round(4)
print(res.to_string(index=False))
res.to_csv("model_comparison.csv", index=False)