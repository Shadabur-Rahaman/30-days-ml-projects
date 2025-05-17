# Titanic Survival Prediction - Day 3

## 📝 Project Overview
A machine learning pipeline to predict passenger survival on the Titanic dataset. Day 3 covers:
- Data loading & inspection
- Missing value imputation
- Categorical encoding
- Feature scaling
- Model training (Random Forest)
- Generating Kaggle submission

## 📂 Repository Structure
```text
titanic-survival-prediction-day3/
├── data/
│   ├── train.csv             # Original training data
│   ├── test.csv              # Original test data
│   └── submission.csv        # Output for Kaggle
├── notebooks/
│   └── day3_data_processing_and_modeling.ipynb  # End-to-end EDA & modeling
├── src/
│   ├── data_preprocessing.py # Data cleaning & feature functions
│   ├── model.py              # Training & prediction pipeline
│   └── utils.py              # Helper functions
├── requirements.txt          # Python dependencies
├── README.md                 # This file
└── .gitignore                # Untracked files config
