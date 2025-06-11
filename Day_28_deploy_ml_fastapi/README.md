

---

# 🚀 Day 28: ML Model Deployment with FastAPI

## 🧠 Project Title

**Deploying a Scikit-learn Model with FastAPI for Real-time Predictions**

---

## 🔍 Overview

This project demonstrates how to train, deploy, and serve predictions from a machine learning model using **FastAPI**, a modern, high-performance Python web framework. The model is trained with **scikit-learn**, and input validation is handled using **Pydantic**.

We expose a REST API endpoint `/predict` that accepts JSON input and returns model predictions in real-time. The API is testable through Swagger UI or Python scripts.

---

## 🗂️ Directory Structure

```bash
Day28_Deploy_ML_FastAPI/
├── model/
│   └── model.pkl                         # Trained classifier model (saved with joblib)
│
├── notebooks/
│   └── Day_28_deploy_ml_fastapi.ipynb    # Main notebook demonstrating training + API usage
│
├── images/
│   └── api_predictions.png               # Screenshot of prediction via API
│
├── src/
│   ├── main.py                           # FastAPI app with /predict endpoint
│   ├── schema.py                         # Pydantic input model for request validation
│   ├── predict.py                        # Logic for loading model & predicting
│   ├── train.py                          # Script to train and save model
│
├── test/
│   └── test_api.py                       # Script to test the API locally
│
├── outputs/
│   ├── sentiment_model.joblib
│   └── Dockerfile                        # Optional: containerize the app
│
├── requirements.txt                      # List of required Python packages
├── .gitignore                            # Ignore virtualenvs, __pycache__, models, etc.
└── README.md                             # Project documentation
```

---

## ⚙️ How to Run

### ✅ 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### ✅ 2. Train the Model

```bash
python src/train.py
```

This creates the `model/model.pkl` file.

### ✅ 3. Start the FastAPI Server

```bash
uvicorn src.main:app --reload
```

Access Swagger UI for testing: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### ✅ 4. Test the API

You can run the test script:

```bash
python test/test_api.py
```

Or use `curl`:

```bash
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d '{"feature1": 5.1, "feature2": 3.5, "feature3": 1.4, "feature4": 0.2}'
```

---

## 🔢 Sample Input

```json
{
  "feature1": 5.1,
  "feature2": 3.5,
  "feature3": 1.4,
  "feature4": 0.2
}
```

## 🎯 Sample Output

```json
{
  "prediction": "setosa"
}
```

---

## ✅ Features

* 📦 FastAPI server with `/predict` endpoint
* 🔐 Input validation using **Pydantic**
* 🧠 ML model trained using **scikit-learn** (e.g., Logistic Regression)
* 🌐 Auto-generated Swagger UI
* 🔁 Real-time predictions using JSON payload
* 🧪 Testable with Python or curl

---

## 🎓 Learning Outcomes

* ✅ Deploy ML models with FastAPI
* ✅ Use Pydantic for input validation
* ✅ Serve predictions via REST API
* ✅ Test and debug real-time API responses

---

## 🧵 Stay Tuned

This is **Day 28** of my **#30DaysMLProjects** challenge.

### 📅 Coming up Next (Day 29):

📈 **Explainable AI** – Using **SHAP & LIME** to interpret black-box ML models!

---

Let me know if you want the **GitHub repo structure** with `README.md`, `.gitignore`, and code pre-filled — I can generate that next!
