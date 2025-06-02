# 🎬 Day 19 – Collaborative Filtering Recommender System

This project demonstrates how to build a movie recommender system using **Collaborative Filtering**.

## 🔍 Objective
To recommend movies based on user behavior using:
- **User-Based Collaborative Filtering**
- **Item-Based Collaborative Filtering**

---

## 🛠️ Technologies & Libraries
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `surprise` (for advanced recommendation models)

---
## 🧰 File Structure
```bash
Day19_Recommender_CollaborativeFiltering_Cleaned/
├── notebooks/
│   └── Day19_Recommender_CollaborativeFiltering_Cleaned.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── model_collaborative.py
│   └── evaluation.py
├── outputs/
│   ├── user_recommendations_sample.csv
│   ├── similarity_matrix.png
│   └── performance_metrics.txt
├── README.md
├── requirements.txt
└── .gitignore

---

🔧 How to Run

1. Install requirements:
pip install -r requirements.txt
2. Run Jupyter Notebook
Open notebooks/Day19_Recommender_CollaborativeFiltering_Cleaned.ipynb to explore the full workflow.

📌 Key Concepts
User-Based CF: Recommend items liked by similar users.

Item-Based CF: Recommend similar items to ones user has liked.

Similarity Metrics: Cosine Similarity, Pearson Correlation

Evaluation: RMSE, MAE, Precision@k, Recall@k

📈 Outputs
✅ Top-N movie recommendations

📊 Similarity heatmap

📄 Performance metrics

✍️ Author
Shadabur Rahaman
🔗 GitHub

