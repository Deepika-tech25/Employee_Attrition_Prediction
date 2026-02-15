import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


print("### Data Preprocessing ###")

# Loading dataset
df = pd.read_csv("dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv")

# convert target column
df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

print("Df Shape (Before removing):",df.shape)

# removing useless columns
df = df.drop(['EmployeeNumber', 'EmployeeCount', 'Over18', 'StandardHours'], axis=1)

print("Df Shape (After removing):",df.shape)

# split the features and target
X = df.drop('Attrition', axis=1)
y = df['Attrition']

# convert the categorical columns
X = pd.get_dummies(X, drop_first=True)

print("Features shape:", X.shape)
print("Target shape:", y.shape)


# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=41
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# save scaler
joblib.dump(scaler, "models/scaler.pkl")

print("Training samples:", X_train.shape)
print("Testing samples:", X_test.shape)

joblib.dump(X_test, "models/X_test.pkl")
joblib.dump(y_test, "models/y_test.pkl")

print("Test data saved successfully")

all_metrics = {}

print("### Logistic Regression Model Training and Prediction ###")
# Logistic Regression model

# Create model
log_model = LogisticRegression(max_iter=2000)

# Train model
log_model.fit(X_train, y_train)

print("Logistic Regression model trained")

# Make predictions
y_pred = log_model.predict(X_test)
y_prob = log_model.predict_proba(X_test)[:, 1]

print("Logistic Regression prediction completed")

# Calculate metrics
log_accuracy = accuracy_score(y_test, y_pred)
log_auc = roc_auc_score(y_test, y_prob)
log_precision = precision_score(y_test, y_pred)
log_recall = recall_score(y_test, y_pred)
log_f1 = f1_score(y_test, y_pred)
log_mcc = matthews_corrcoef(y_test, y_pred)

# Save results
log_metrics = {
    "Accuracy": log_accuracy,
    "AUC": log_auc,
    "Precision": log_precision,
    "Recall": log_recall,
    "F1 Score": log_f1,
    "MCC": log_mcc
}

print("Logistic Regression Metrics:")
print(log_metrics)

# Save model
joblib.dump(log_model, "models/logistic_regression.pkl")

all_metrics["Logistic Regression"] = log_metrics

print("### Logistic Regression model saved successfully ###")

print("### Decision Tree Model Training and Prediction ###")

dt_model = DecisionTreeClassifier(random_state=41)

dt_model.fit(X_train, y_train)

print("Decision Tree model trained")

y_pred_dt = dt_model.predict(X_test)
y_prob_dt = dt_model.predict_proba(X_test)[:, 1]

print("Decision Tree prediction completed")

dt_accuracy = accuracy_score(y_test, y_pred_dt)
dt_auc = roc_auc_score(y_test, y_prob_dt)
dt_precision = precision_score(y_test, y_pred_dt)
dt_recall = recall_score(y_test, y_pred_dt)
dt_f1 = f1_score(y_test, y_pred_dt)
dt_mcc = matthews_corrcoef(y_test, y_pred_dt)

dt_metrics = {
    "Accuracy": dt_accuracy,
    "AUC": dt_auc,
    "Precision": dt_precision,
    "Recall": dt_recall,
    "F1 Score": dt_f1,
    "MCC": dt_mcc
}

print("Decision Tree Metrics:")
print(dt_metrics)

all_metrics["Decision Tree"] = dt_metrics

joblib.dump(dt_model, "models/decision_tree.pkl")

print("### Decision Tree model saved successfully ###")

print("### KNN Model Training and Prediction ###")

knn_model = KNeighborsClassifier(n_neighbors=5)

knn_model.fit(X_train, y_train)

print("KNN model trained")

y_pred_knn = knn_model.predict(X_test)
y_prob_knn = knn_model.predict_proba(X_test)[:, 1]

print("KNN prediction completed")

knn_accuracy = accuracy_score(y_test, y_pred_knn)
knn_auc = roc_auc_score(y_test, y_prob_knn)
knn_precision = precision_score(y_test, y_pred_knn)
knn_recall = recall_score(y_test, y_pred_knn)
knn_f1 = f1_score(y_test, y_pred_knn)
knn_mcc = matthews_corrcoef(y_test, y_pred_knn)

knn_metrics = {
    "Accuracy": knn_accuracy,
    "AUC": knn_auc,
    "Precision": knn_precision,
    "Recall": knn_recall,
    "F1 Score": knn_f1,
    "MCC": knn_mcc
}

print("KNN Metrics:")
print(knn_metrics)

all_metrics["KNN"] = knn_metrics

joblib.dump(knn_model, "models/knn.pkl")

print("### KNN model saved successfully ###")

print("### Naive Bayes Model Training and Prediction ###")

nb_model = GaussianNB()

nb_model.fit(X_train, y_train)

print("Naive Bayes model trained")

y_pred_nb = nb_model.predict(X_test)
y_prob_nb = nb_model.predict_proba(X_test)[:, 1]

print("Naive Bayes prediction completed")

nb_accuracy = accuracy_score(y_test, y_pred_nb)
nb_auc = roc_auc_score(y_test, y_prob_nb)
nb_precision = precision_score(y_test, y_pred_nb)
nb_recall = recall_score(y_test, y_pred_nb)
nb_f1 = f1_score(y_test, y_pred_nb)
nb_mcc = matthews_corrcoef(y_test, y_pred_nb)

nb_metrics = {
    "Accuracy": nb_accuracy,
    "AUC": nb_auc,
    "Precision": nb_precision,
    "Recall": nb_recall,
    "F1 Score": nb_f1,
    "MCC": nb_mcc
}

print("Naive Bayes Metrics:")
print(nb_metrics)

all_metrics["Naive Bayes"] = nb_metrics

joblib.dump(nb_model, "models/naive_bayes.pkl")

print("### Naive Bayes model saved successfully ###")

print("### Random Forest Model Training and Prediction ###")

rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=41,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

print("Random Forest model trained")

y_pred_rf = rf_model.predict(X_test)
y_prob_rf = rf_model.predict_proba(X_test)[:, 1]

print("Random Forest prediction completed")

rf_accuracy = accuracy_score(y_test, y_pred_rf)
rf_auc = roc_auc_score(y_test, y_prob_rf)
rf_precision = precision_score(y_test, y_pred_rf)
rf_recall = recall_score(y_test, y_pred_rf)
rf_f1 = f1_score(y_test, y_pred_rf)
rf_mcc = matthews_corrcoef(y_test, y_pred_rf)

rf_metrics = {
    "Accuracy": rf_accuracy,
    "AUC": rf_auc,
    "Precision": rf_precision,
    "Recall": rf_recall,
    "F1 Score": rf_f1,
    "MCC": rf_mcc
}

print("Random Forest Metrics:")
print(rf_metrics)

all_metrics["Random Forest"] = rf_metrics

joblib.dump(rf_model, "models/random_forest.pkl")

print("### Random Forest model saved successfully ###")

print("### XGBoost Model Training and Prediction ###")

xgb_model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    random_state=41,
    use_label_encoder=False,
    eval_metric='logloss'
)

xgb_model.fit(X_train, y_train)

print("XGBoost model trained")

y_pred_xgb = xgb_model.predict(X_test)
y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]

print("XGBoost prediction completed")

xgb_accuracy = accuracy_score(y_test, y_pred_xgb)
xgb_auc = roc_auc_score(y_test, y_prob_xgb)
xgb_precision = precision_score(y_test, y_pred_xgb)
xgb_recall = recall_score(y_test, y_pred_xgb)
xgb_f1 = f1_score(y_test, y_pred_xgb)
xgb_mcc = matthews_corrcoef(y_test, y_pred_xgb)

xgb_metrics = {
    "Accuracy": xgb_accuracy,
    "AUC": xgb_auc,
    "Precision": xgb_precision,
    "Recall": xgb_recall,
    "F1 Score": xgb_f1,
    "MCC": xgb_mcc
}

print("XGBoost Metrics:")
print(xgb_metrics)

all_metrics["XGBoost"] = xgb_metrics

joblib.dump(xgb_model, "models/xgboost.pkl")

print("### XGBoost model saved successfully ###")

print("\n### FINAL MODEL COMPARISON ###")

comparison_df = pd.DataFrame(all_metrics).T
print(comparison_df)




