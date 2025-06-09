# 🧠 Multi-Task Learning with PyTorch

This project demonstrates how to train a **Multi-Task Neural Network** using PyTorch that can perform:

- **Task 1:** Binary Classification (e.g., predict whether a customer will churn)  
- **Task 2:** Regression (e.g., predict how much a customer will spend)

---

## 🔎 Overview

**Multi-task learning** allows one model to learn **multiple objectives simultaneously** by sharing representations across tasks. This improves model performance, generalization, and efficiency, especially when tasks are related.

### 🧩 Key Concepts:
- ✅ Shared base layers for feature learning  
- ✅ Task-specific heads for classification and regression  
- ✅ Combined weighted loss (BCE + MSE)  

---
### 📚 Learning Outcomes
By completing this project, you will:

✅ Understand multi-task learning fundamentals
✅ Build shared + task-specific heads in PyTorch
✅ Use joint loss functions (Binary Cross-Entropy + MSE)
✅ Visualize performance per task
✅ Train efficient models that handle multiple objectives

---
## 🗂️ Directory Structure

```bash
Day26_Multi_Task_Learning_Cleaned/
├── data/
│   └── customers.csv                     # Sample tabular dataset with features and labels
├── images/
│   ├── history.png                       # Training loss curves
│   └── prediction.png                    # Actual vs predicted regression outputs
├── notebooks/
│   └── Day_26_Multi_task_Learning.ipynb # Main notebook demonstrating multi-task training
├── src/
│   ├── model.py                         # Multi-task model class
│   ├── dataset.py                       # Dataset loader and preprocessor
│   ├── train.py                         # Training and evaluation loop
│   └── utils.py                         # Plotting and evaluation utilities
├── requirements.txt                     # Required Python packages
├── .gitignore                           # Common file ignores
└── README.md                            # Project documentation
⚙️ Model Architecture

Input Features
     │
     ▼
[ Shared Layers ]
     │
 ┌───┴────────┐
▼            ▼
Classification Head   Regression Head
(Sigmoid)             (Linear)
📈 Visualizations
📉 history.png 

📊 prediction.png 

🧪 How to Run
🔧 Install Dependencies
pip install -r requirements.txt

📓 Run Notebook
jupyter notebook notebooks/Day_26_Multi_task_Learning.ipynb

🖥️ Run Python Script (Optional)

python src/train.py




