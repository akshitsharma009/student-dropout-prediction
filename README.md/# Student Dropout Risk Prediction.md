🎓 Student Dropout Risk Prediction System
📌 Project Overview

Student dropout is a critical challenge for educational institutions and often goes unnoticed until it is too late.
This project builds an end-to-end Machine Learning system that predicts student dropout risk using academic, behavioral, and support-related data.

The system goes beyond basic classification by using probability-based prediction, custom thresholds, and risk categorization to enable early and actionable intervention.

🎯 Problem Statement

Manual monitoring of student performance does not scale well and makes early identification of at-risk students difficult, especially with large student populations.

The challenge is to:

Detect dropout risk early

Minimize missed at-risk students

Avoid over-flagging safe students

✅ Objectives

Analyze student academic and behavioral data

Train multiple machine learning models to predict dropout risk

Optimize recall for at-risk students

Deploy the final model via a backend API for real-world usage

💡 Why This Project Matters

Early identification of at-risk students allows institutions to:

Improve retention rates

Provide timely academic support

Reduce long-term academic and financial loss

This project demonstrates how machine learning can support decision-making, not just prediction.

📊 Exploratory Data Analysis & Model Training
Models Evaluated

The following models were trained and evaluated:

Logistic Regression

Balanced Logistic Regression

Decision Tree

Random Forest

Final Model Selection

Balanced Logistic Regression was selected as the final model.

Reason:

Delivered the best recall for at-risk students

Reduced the number of missed dropout cases

Provided interpretable probability outputs

📈 Evaluation Metric
Why Recall over Accuracy?

In dropout prediction, missing an at-risk student is more costly than flagging a safe one.

Therefore:

Recall was prioritized over accuracy

The objective was to minimize false negatives

This aligns better with real-world academic intervention scenarios

🎯 Threshold Tuning & Decision Strategy

The default probability threshold of 0.5 was found to be suboptimal due to a high number of missed at-risk students.

Improvements Achieved

Recall improved from 62% to 88%

False negatives were significantly reduced

Final Threshold Strategy

Although threshold experimentation identified 0.35 as optimal during training, the production system uses:

Business Threshold: 0.6

This avoids over-flagging borderline students

Enables controlled, explainable decisions

🚦 Risk Categorization (Production Logic)

Instead of a binary output, predictions are grouped into risk buckets:

Probability Range	Risk Level
< 0.4	Low Risk
0.4 – 0.6	Medium Risk
≥ 0.6	High Risk

This allows institutions to:

Monitor medium-risk students

Prioritize high-risk cases

Avoid unnecessary intervention

🔧 Model Persistence & Deployment Readiness

The trained model was serialized using joblib

Backend inference uses the same feature schema as training

Threshold logic is configurable at inference time

REST API built using Flask

🗂 Project Structure
student-dropout-prediction/
│
├── backend/
│   ├── app.py
│   └── utils/
│       └── preprocess.py
│
├── models/
│   └── dropout_model.pkl
│
├── notebooks/
│   └── model_training_and_evaluation.ipynb
│
├── data/
│   └── dataset.csv
│
├── README.md
└── requirements.txt

🔌 API Example
Request
{
  "studytime": 2,
  "failures": 0,
  "absences": 4,
  "schoolsup": 1,
  "famsup": 1,
  "paid": 0,
  "higher": 1,
  "internet": 1
}

Response
{
  "prediction": "No Dropout",
  "probability": 0.548,
  "risk_level": "Medium Risk",
  "threshold_used": 0.6
}

🧠 What This Project Demonstrates

Strong understanding of classification metrics

Real-world threshold tuning

End-to-end ML deployment mindset

Backend integration for ML inference

Decision-focused ML system design

🏁 Conclusion

This project moves beyond academic modeling and demonstrates how machine learning can be applied responsibly in real educational environments.
By combining probability, thresholds, and risk categorization, the system enables practical and explainable intervention.

👨‍💻 Author

Akshit Sharma
B.Tech | Data Science & Machine Learning
GitHub: (https://github.com/akshitsharma009)
LinkedIn: (https://www.linkedin.com/in/akshit-sharma-7427362a0)