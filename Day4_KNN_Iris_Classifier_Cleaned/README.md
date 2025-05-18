
# 🌸 Day 4: K-Nearest Neighbors Classifier on Iris Dataset

This repository contains a Jupyter Notebook demonstrating the use of **K-Nearest Neighbors (KNN)** for classifying species in the classic **Iris dataset**.

## 📘 Project Overview

- Perform data exploration and visualization
- Preprocess features using `StandardScaler`
- Train and evaluate a KNN model
- Analyze classification results and metrics

## 📁 Repository Structure

Day4-KNN-Iris-Classifier/
├── data/ # (Optional) Place for iris dataset (if downloaded separately)
├── notebooks/
│ └── Day4_KNN_Iris_Classifier_Cleaned.ipynb
├── README.md
└── .gitignore



## 🚀 How to Run

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Day4-KNN-Iris-Classifier.git
cd Day4-KNN-Iris-Classifier
(Optional) Create a virtual environment:

bash
Copy
Edit
python -m venv venv
source venv/bin/activate  # Windows: venv\\Scripts\\activate
Install required packages:

bash
Copy
Edit
pip install -r requirements.txt
Launch the notebook:

bash
Copy
Edit
jupyter notebook notebooks/Day4_KNN_Iris_Classifier_Cleaned.ipynb
🛠 Dependencies
Make sure you have the following Python packages installed:

pandas

numpy

seaborn

matplotlib

scikit-learn

jupyter

You can install them using:

bash
Copy
Edit
pip install pandas numpy seaborn matplotlib scikit-learn jupyter
📊 Output
The notebook includes:

Pair plots for visual EDA

KNN classifier training

Confusion matrix & classification report

Cross-validation performance

