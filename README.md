# Heart Disease Prediction: ML Assignment 2

## a. Problem statement
The goal is to predict whether a patient has heart disease from 13 clinical
measurements (age, blood pressure, cholesterol, chest pain type and others).
Five classification models are trained on the same dataset, compared on six
evaluation metrics, and served through a Streamlit web app that scores an
uploaded test CSV with the saved models.

## b. Dataset description
- Source: Kaggle, "Heart Disease Dataset" (johnsmith88)
- Instances: 1025; Features: 13; Target: `target` (1 = heart disease, 0 = no disease)
- Classes are balanced (526 vs 499)
- Features: age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang,
  oldpeak, slope, ca, thal
- Preprocessing: StandardScaler on numeric columns, one-hot encoding on
  categorical columns, 80/20 stratified train/test split (random_state=2026)
- Limitation: the file contains many duplicate rows (about 302 unique
  records), so some test rows also appear in training and scores may be optimistic.

## c. GitHub repository
https://github.com/varshaburra10/heart-disease-ml-app

Live app: https://heart-disease-ml-app-o9rkummbphrmgku8n6sula.streamlit.app/

## d. Models used

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8927 | 0.9491 | 0.8879 | 0.9048 | 0.8962 | 0.7853 |
| Decision Tree | 0.9171 | 0.9754 | 0.8793 | 0.9714 | 0.9231 | 0.8385 |
| KNN | 0.8683 | 0.9451 | 0.8980 | 0.8381 | 0.8670 | 0.7386 |
| Naive Bayes | 0.8488 | 0.9009 | 0.8491 | 0.8571 | 0.8531 | 0.6973 |
| Support Vector Machine | 0.9073 | 0.9721 | 0.9300 | 0.8857 | 0.9073 | 0.8157 |

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Strong linear baseline with balanced precision and recall (MCC 0.785); the classes are largely separable by a linear boundary. |
| Decision Tree | Best accuracy, AUC, F1 and MCC, with very high recall (0.971), so it misses few patients; precision is lower, so it raises some false alarms. Duplicate rows may inflate this result. |
| KNN | Good precision (0.898) but lower recall (0.838), so it misses more diseased patients; depends on distances after scaling and encoding. |
| Naive Bayes | Weakest on every metric (MCC 0.697); its feature-independence assumption is likely violated because the clinical features are correlated. |
| Support Vector Machine | Highest precision (0.930) with AUC and MCC close to the tree; the RBF kernel captures non-linear patterns while generalising smoothly. |
| **Overall winner** | Decision Tree on most metrics (MCC 0.8385, recall 0.971); SVM is a close second and may generalise better given the duplicate rows. |
