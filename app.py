import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)



st.set_page_config(page_title="Employee Attrition Predictor", layout="wide")

st.title("Employee Attrition Prediction System")

# =============================
# Load scaler and test data
# =============================
scaler = joblib.load("models/scaler.pkl")
X_test = joblib.load("models/X_test.pkl")
y_test = joblib.load("models/y_test.pkl")

# =============================
# Model selection
# =============================
st.sidebar.header("Model Selection")

model_map = {
    "Logistic Regression": "models/logistic_regression.pkl",
    "Decision Tree": "models/decision_tree.pkl",
    "KNN": "models/knn.pkl",
    "Naive Bayes": "models/naive_bayes.pkl",
    "Random Forest": "models/random_forest.pkl",
    "XGBoost": "models/xgboost.pkl"
}

model_choice = st.sidebar.selectbox(
    "Choose a trained model",
    list(model_map.keys())
)

model = joblib.load(model_map[model_choice])

st.sidebar.success(f"{model_choice} loaded")

# =============================
# Display evaluation metrics
# =============================
st.header("Model Evaluation Metrics")

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
mcc = matthews_corrcoef(y_test, y_pred)

metrics_df = pd.DataFrame({
    "Metric": ["Accuracy", "AUC", "Precision", "Recall", "F1 Score", "MCC"],
    "Value": [accuracy, auc, precision, recall, f1, mcc]
})

st.dataframe(metrics_df, use_container_width=True)

# =============================
# Confusion Matrix
# =============================
st.header("Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots()

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No Attrition", "Attrition"],
    yticklabels=["No Attrition", "Attrition"],
    ax=ax
)

ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title(f"{model_choice} Confusion Matrix")

st.pyplot(fig)

# =============================
# Classification Report
# =============================
st.header("Classification Report")

report = classification_report(y_test, y_pred, output_dict=True)

report_df = pd.DataFrame(report).transpose()

st.dataframe(report_df, use_container_width=True)

# =============================
# Disclaimer
# =============================
st.warning("""
IMPORTANT DISCLAIMER:

This model was trained on IBM HR Attrition dataset.
Upload dataset must follow same schema for accurate results.
""")

# =============================
# Prediction Section
# =============================
st.header("Upload New Dataset for Prediction")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    try:

        original_df = df.copy()

        drop_cols = [
            'EmployeeNumber',
            'EmployeeCount',
            'Over18',
            'StandardHours'
        ]

        df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors='ignore')

        df = pd.get_dummies(df, drop_first=True)

        training_columns = scaler.feature_names_in_

        missing_cols = set(training_columns) - set(df.columns)

        for col in missing_cols:
            df[col] = 0

        df = df[training_columns]

        df_scaled = scaler.transform(df)

        predictions = model.predict(df_scaled)
        probabilities = model.predict_proba(df_scaled)[:, 1]

        original_df["Attrition Prediction"] = [
            "Yes" if p == 1 else "No"
            for p in predictions
        ]

        original_df["Attrition Probability"] = probabilities

        st.subheader("Prediction Results")
        st.dataframe(original_df)

        csv = original_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Download Results",
            csv,
            "predictions.csv",
            "text/csv"
        )

        st.success("Prediction completed")

    except Exception as e:
        st.error(str(e))

