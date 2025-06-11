🚀 ML Model Deployment with FastAPI (Day 28 of #30DaysMLProjects)
This project demonstrates how to train, deploy, and test a machine learning model using FastAPI, a modern, fast (high-performance) Python web framework for building APIs.

🔍 Project Overview
We build a basic machine learning pipeline for classifying data, train a model using scikit-learn, and deploy it via a RESTful API using FastAPI. Input validation is handled with pydantic. The API can serve real-time predictions and is testable using requests.

📁 Directory Structure
```bash
Day28_Deploy_ML_FastAPI/
├── model/
│   └── model.pkl                         # Trained classifier model (saved with joblib)
│
├── notebooks/
│   └── Day_28_deploy_ml_fastapi.ipynb    # Main notebook demonstrating training + API usage
│
├── images/
│   └── api_predictions.png 
│
├── src/
│   ├── main.py                           # FastAPI app with /predict endpoint
│   ├── schema.py                         # Pydantic input model
│   ├── predict.py                        # Load model and make predictions
│   ├── train.py                          # Train & save model script
│  
├── test/
│    └── test_api.py                       # Send request to FastAPI for testing
│
├── outputs/
│   └── sentiment_model.joblib
│   └── Dockerfile
│
├── requirements.txt                      # Python dependencies
├── .gitignore                            # Ignore model files, __pycache__, etc.
└── README.md                             # You're here!

⚙️ How to Run
1. Install Dependencies
pip install -r requirements.txt
2. Train the Model

python src/train.py
This will create model/model.pkl.

3. Start the API

uvicorn src.main:app --reload
Visit: http://127.0.0.1:8000/docs for Swagger UI to test the endpoint.

4. Test with curl or Python

python src/test_api.py
✅ Features
📦 FastAPI server with /predict endpoint

✅ Input validation using pydantic

🧠 Scikit-learn model: e.g., Logistic Regression

🔁 Live prediction using JSON payload

📊 Swagger UI auto-generated for testing

🔍 Sample Input

{
  "feature1": 5.1,
  "feature2": 3.5,
  "feature3": 1.4,
  "feature4": 0.2
}
🧠 Sample Output
{
  "prediction": "setosa"
}
📌 Learning Outcomes
✅ Deploy machine learning models using FastAPI
✅ Use Pydantic for type-safe input validation
✅ Build production-ready REST APIs
✅ Run, test, and evaluate real-time predictions

🧵 Stay Tuned
This is Day 28 of my #30DaysMLProjects series.

📅 Coming up next (Day 29):
📈 Explainable AI (SHAP & LIME) — Visualize how features impact predictions!
