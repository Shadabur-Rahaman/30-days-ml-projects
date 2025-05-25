# CNN Classifier on CIFAR-10 Dataset

This project implements a **Convolutional Neural Network (CNN)** using Keras to classify images from the **CIFAR-10** dataset. It is part of my #30DaysMLProjects initiative.

![CNN CIFAR-10](images/cifar10_cnn_arch.png)  
*A simple CNN architecture built with Keras for image classification*

## 🚀 Features

- CIFAR-10 dataset (10 classes of 32x32 color images)
- CNN architecture with Conv2D, MaxPooling2D, Dropout, Dense
- Accuracy tracking and evaluation
- TensorFlow/Keras-based implementation

## 📦 Usage
## 🔧 How to Run
1. Install dependencies:
```bash
\
📁 Folder Structure
Day_11_CNN_CIFAR10_Classifier_Cleaned/
│
├── notebooks
|   └── Day11_CNN_CIFAR10_Classifier_Cleaned.ipynb  # Main notebook for CNN on CIFAR-10
│
├── src/                            
|   └── cnn_utils.py                                # Utility functions for preprocessing, model building
│
├── images/                                         # Visual outputs (optional)
│   ├── training_accuracy_loss_curve.png
│   └── cifar10_predictions_grid.png
│
├── requirements.txt                                # Libraries: tensorflow, matplotlib, seaborn, numpy
├── .gitignore                                      # __pycache__/, .ipynb_checkpoints/, etc.
└── README.md                                       # Project overview, setup, results, and learnings

1. Clone the repo:
   git clone https://github.com/Shadabur-Rahaman/30-days-ml-projects.git
   cd Day_11_CNN_CIFAR10_Classifier_Cleaned

Install dependencies:

pip install -r requirements.txt

Run the notebook or script to train the model and evaluate accuracy.
jupyter notebook notebooks/Day11_CNN_CIFAR10_Classifier_Cleaned.ipynb

🧠 Learning Outcomes
Understand CNN building blocks (Conv, Pool, Flatten, Dense)

Use TensorFlow/Keras to build deep learning models

Work with image data and visualize performance
