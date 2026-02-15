```md
Employee Attrition Prediction System

Live Streamlit App
Click below URL to use the app
```
https://employeeattritionprediction-2025aa05609-deepika.streamlit.app/
```
---
Problem Statement:

Employee attrition is a major challenge faced by organizations as it leads to increased hiring costs, loss of experienced employees, and reduced productivity. The goal of this project is to build a Machine Learning system that can predict whether an employee is likely to leave the organization based on HR-related features such as job role, salary, work experience, job satisfaction, and work environment.
This system helps HR departments take proactive steps to retain employees.

---
Dataset Description:
Dataset used: IBM HR Employee Attrition Dataset [You can find the dataset in below link]
```
https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset
```
- Number of records: 1470 employees
- Number of features: 35
- Target variable: Attrition (Yes/No)

Target variable description:
Attrition = Yes (Employee leaves)
Attrition = No (Employee stays)

---
Machine Learning Models Used:
The following classification models were trained and evaluated:
1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Naive Bayes
5. Random Forest
6. XGBoost

---
Model Evaluation Metrics
The models were evaluated using:
- Accuracy
- AUC Score
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

---
Model Comparison Results

| Model                | Accuracy | AUC   | Precision | Recall | F1 Score | MCC   |
|----------------------|----------|-------|-----------|--------|----------|-------|
| Logistic Regression  | 0.891    | 0.834 | 0.625     | 0.500  | 0.556    | 0.498 |
| Decision Tree        | 0.779    | 0.598 | 0.264     | 0.350  | 0.301    | 0.175 |
| KNN                  | 0.867    | 0.691 | 0.529     | 0.225  | 0.316    | 0.284 |
| Naive Bayes          | 0.714    | 0.819 | 0.292     | 0.775  | 0.425    | 0.343 |
| Random Forest        | 0.895    | 0.830 | 0.909     | 0.250  | 0.392    | 0.445 |
| XGBoost              | 0.912    | 0.835 | 0.889     | 0.400  | 0.552    | 0.561 |

---
Observations:
Logistic Regression
Logistic Regression performed well with an accuracy of 89.11% and good overall balance between precision and recall. It has strong generalization and stability. The MCC score of 0.498 indicates reliable prediction performance. 

Decision Tree
Decision Tree showed lower performance compared to other models with accuracy of 77.89% and MCC of 0.175. This suggests overfitting and poor generalization to unseen data.

K-Nearest Neighbors (KNN)
KNN achieved good accuracy (86.73%) but low recall (22.5%) means it failed to identify many employees who would leave. This makes it less suitable when identifying attrition risk which is important.

Naive Bayes
Naive Bayes had lower accuracy (71.43%) but very high recall (77.5%) means it identified most employees likely to leave. But the precision is low means it gives more false positives.

Random Forest
Random Forest achieved strong accuracy (89.45%) and excellent precision (90.9%) means its positive predictions were highly reliable. Here the recall is low indicating it missed some attrition cases. 

XGBoost
XGBoost achieved the best overall performance with highest accuracy (91.15%) and highest MCC (0.5607). It provided the best balance between precision, recall, and overall predictive power.

---
Best Performing Model:
Based on evaluation metrics, XGBoost performed the best with:
- Highest Accuracy: 91.15%
- Highest MCC: 0.5607
- Best overall performance balance
Therefore, XGBoost is the most effective model for predicting employee attrition.

---
Application Features
The Streamlit web application allows users to:
- Select different trained models
- View model evaluation metrics
- View confusion matrix
- View classification report
- Upload new employee dataset
- Predict employee attrition
- View prediction probability
- Download prediction results

---
Technologies Used

- Python
- Scikit-learn
- XGBoost
- Pandas
- NumPy
- Streamlit
- Matplotlib
- Seaborn
- Joblib
---
To Run Locally
Step 1: Clone repository
```

git clone https://github.com/your-username/employee-attrition-predictor.git

```
Step 2: Install dependencies
```

pip install -r requirements.txt

```
Step 3: Run Streamlit app
```

streamlit run app.py

```
---
Project Structure

employee-attrition-predictor/
│
├── app.py
├── train_models.py
├── requirements.txt
├── README.md
├── models/
├── dataset/

---
Conclusion

This project successfully developed and deployed a Machine Learning system to predict employee attrition. Multiple models were trained and evaluated, and XGBoost was found to be the best performing model. The system was deployed using Streamlit, allowing users to interactively upload datasets and obtain predictions.

This solution can help organizations identify at-risk employees and improve retention strategies.
```




