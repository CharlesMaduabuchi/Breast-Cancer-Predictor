# 🩺 Breast Cancer Prediction Web App

This is a machine learning-powered web application built with **Flask** that predicts the likelihood of breast cancer based on user-provided diagnostic features. The model is trained using **XGBoost** and deployed via a user-friendly interface.

---

## 📊 Features

- 🔍 Predict breast cancer: Cancerous or Non-Cancerous
- 💡 Powered by an XGBoost classification model
- 🌐 Responsive frontend using HTML, CSS, and Remix Icons
- ⚙️ JSON API endpoint for programmatic access (`/predict`)
- 📥 Supports both form submission and AJAX-based predictions
- 🔐 CORS enabled for cross-origin requests

---

## 🚀 Demo

> Coming soon! You can deploy it on [Render](https://render.com), [Railway](https://railway.app), or AWS EC2.

---

## 🧠 Model Details

- Algorithm: `XGBoostClassifier`
- Dataset: Breast Cancer Wisconsin (Diagnostic) Dataset
- Features used:
  - `radius_mean`
  - `texture_mean`
  - `perimeter_mean`
  - `area_mean`
  - `smoothness_mean`
  - `compactness_mean`
  - `concavity_mean`
  - `concave_point_mean`

---

## 🛠️ Tech Stack

- **Backend:** Flask, Flask-CORS, NumPy, joblib
- **Frontend:** HTML5, CSS3, JavaScript, 
- **Model:** XGBoost
- **Deployment Ready:** Can be hosted on AWS, Heroku, Render, or Railway

---

## 📂 Project Structure

breast-cancer-prediction/
├── app.py
├── xgb_breast_cancer_model.pkl
├── requirements.txt
├── templates/
│ └── index.html
├── static/
│ ├── styles.css
│ └── assets/
│ ├── about.jpg
│ └── doctor-1.jpg (and others)
└── README.md


📦 Requirements
Install dependencies with:
pip install -r requirements.txt
flask
flask-cors
xgboost
joblib
numpy



📄 License
This project is licensed under the MIT License.


🙌 Acknowledgments
UCI Breast Cancer Dataset

XGBoost

Flask Documentation