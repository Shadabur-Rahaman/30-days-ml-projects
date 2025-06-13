# 🧠 Day 30 of #30DaysMLProjects: End-to-End Machine Learning Pipeline

Welcome to the final project of my 30 Days of ML Projects challenge! This project demonstrates the creation of a complete end-to-end machine learning pipeline, encapsulating the full workflow from data ingestion to deployment-ready model packaging.

---

## 🚀 Project Highlights

- **Data Source**: Utilized real-world datasets to simulate production scenarios.
- **Pipeline Components**:
  - Data Loading & Preprocessing
  - Feature Engineering
  - Model Selection & Training
  - Hyperparameter Tuning
  - Evaluation (Accuracy, Precision, Recall, F1)
  - Serialization with `joblib`
  - Deployment-ready structure

---

## 🧰 Technologies Used

- Python (Pandas, Scikit-learn, NumPy, Matplotlib, Seaborn)
- Jupyter Notebook
- Joblib
- Clean, modular pipeline architecture

---

## 📁 Folder Structure
```bash
.
├── api
│   ├── Dockerfile
│   ├── main.py
│   ├── requirements.txt
│   └── __pycache__
│       └── main.cpython-311.pyc
├── data
│   ├── .ipynb_checkpoints
│   ├── iris_raw.csv
│   ├── X_test.csv
│   ├── X_train.csv
│   ├── y_test.csv
│   └── y_train.csv
├── github
│   ├── .ipynb_checkpoints
│   └── workflows
│       └── ml-pipeline.yml
├── mlruns
│   ├── .trash
│   ├── 0
│   │   └── meta.yaml
│   └── 655008300926127301
│       ├── 02bcfd2f006648e991bb8cf920ff8455
│       ├── 064d792a000f40edad55a9dcb346a3f6
│       ├── 47a848b640f44cb6b9e6ac9866e004a8
│       ├── 7af705d3d8dd4963a59af661c7a7473e
│       ├── 94102186f16441e4b31349332486433a
│       ├── a74ea3ad2a054addb967c8dbf7e42215
│       ├── meta.yaml
│       ├── models
│       │   ├── m-40ba706ee0594f82b568ea241e24212f
│       │   ├── m-881c86506c564aa2a957c4437ab4cdd9
│       │   ├── m-9ebc73d55674408097573767ca7b8c64
│       │   ├── m-bfa13ba755d34080afbc1b49cea836de
│       │   ├── m-dc21ce9a2d684b13bceecece01574d95
│       │   └── m-dea5154b647b4290afece9e0de2d46e0
│       └── 02bcfd2f006648e991bb8cf920ff8455
│           ├── artifacts
│           ├── metrics
│           ├── outputs
│           ├── params
│           └── tags
├── models
│   ├── iris_classifier.joblib
│   ├── iris_pipeline.joblib
│   └── iris_model.joblib
├── monitoring
│   ├── cm_logistic_regression.png
│   ├── cm_random_forest.png
│   ├── cm_svm.png
│   └── prometheus.yml
├── outputs
│   ├── correlation_heatmap.png
│   ├── decision_boundaries.png
│   ├── feature_distribution.png
│   └── pair_plot.png
├── sample_data
│   ├── .ipynb_checkpoints
│   ├── anscombe.json
│   ├── california_housing_test.csv
│   ├── california_housing_train.csv
│   ├── mnist_test.csv
│   ├── mnist_train_small.csv
│   └── README.md
├── tests
│   └── test_api.py
├── correlation_heatmap.png
├── decision_boundaries.png
├── feature_distribution.png
├── folder_structure.txt
├── iris_model.joblib
├── iris_pipeline.joblib
└── pair_plot.png

```
## 🚦 How to Run

1. **Install requirements**:
   ```bash
   pip install -r requirements.txt
Run Streamlit app:
  ```bash
streamlit run app.py
```

🧾 Key Features
Interactive UI to upload data

Model training with user-selected hyperparameters

Real-time predictions

Beautiful metrics and visualizations

🎉 Final Thoughts
This wraps up my 30-day ML project challenge. This project ties everything together — theory, practice, and deployment!

📬 Connect with Me
LinkedIn: [www.linkedin.com/in/shadabur-rahaman-1b5703249]

GitHub: [https://github.com/Shadabur-Rahaman/30-days-ml-projects/blob/main/]

