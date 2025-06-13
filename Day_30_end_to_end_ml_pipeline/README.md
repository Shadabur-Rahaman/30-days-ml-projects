# 🎓 Student Performance Prediction App

An end-to-end machine learning application that predicts student exam performance based on key demographic and academic factors using **CatBoost Regressor**, EDA, and a production-ready Flask deployment.

✅ This is **Project 30 of 30** in the #30DaysMLProjects series!

---

## 🗂️ Project Structure
```bash
├── .ebextensions/ # Elastic Beanstalk config
│ └── python.config
├── .github/
│ └── workflows/
│ └── main_studentssperformance3.yml # GitHub Actions CI/CD
├── artifacts/
│ ├── data.csv
│ ├── model.pkl
│ ├── preprocessor.pkl
│ ├── train.csv
│ └── test.csv
├── catboost_info/ # CatBoost training metadata
│ ├── learn_error.tsv
│ ├── time_left.tsv
│ └── catboost_training.json
├── notebook/
│ ├── 1 . EDA STUDENT PERFORMANCE .ipynb
│ └── 2. MODEL TRAINING.ipynb
├── src/
│ ├── components/
│ ├── pipeline/
│ ├── exception.py
│ ├── logger.py
│ ├── utils.py
│ ├── zilenames.txt
│ └── init.py
├── templates/
│ ├── home.html
│ ├── index.html
│ └── zilenames.txt
├── app.py # Flask application
├── requirements.txt # Dependencies
├── setup.py # Packaging
├── .gitignore
└── README.md
```


---

## 📊 Features

- **EDA Notebook:** Deep insights into student behavior and academic patterns
- **Modeling:** CatBoost Regressor trained on transformed feature sets
- **Pipeline:** Clean architecture with logging, exception handling, and modular components
- **Web App:** Flask-based front end with interactive prediction form
- **CI/CD:** GitHub Actions + AWS Elastic Beanstalk config included

---

## 🚀 How to Run Locally

### 🔧 Install Dependencies
```bash
pip install -r requirements.txt
🧠 Train Model
bash
Copy
Edit
python src/pipeline/train_pipeline.py
🌐 Launch Web App
bash
Copy
Edit
python app.py
Then visit: http://localhost:5000/

🔍 Sample Inputs
Gender: Male/Female

Parental Level of Education

Lunch Type

Test Preparation Course

Scores in Math, Reading, Writing

✅ Model Info
Algorithm: CatBoost Regressor

Performance: R² ≈ 0.89

Artifacts: model.pkl, preprocessor.pkl

📦 Deployment
Supports AWS Elastic Beanstalk deployment (.ebextensions/)

GitHub Actions for CI/CD via .github/workflows/

📚 Notebooks
1 . EDA STUDENT PERFORMANCE .ipynb

2. MODEL TRAINING.ipynb

📌 Final Words
This marks the completion of my 30-day ML challenge. From simple regressors to advanced Transformers and full-stack apps, this journey has been a deep dive into the world of machine learning — ending with this complete production-grade project.

🧠 Author
Shadabur Rahaman
💼 LinkedIn: linkedin.com/in/shadabur-rahaman
