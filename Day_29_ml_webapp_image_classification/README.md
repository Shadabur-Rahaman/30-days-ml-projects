# 🖼️ Day 29: ML WebApp - Image Classification

## 🧠 Project Title
**Real-Time Image Classification Web App using Flask and CNN**

---

## 🔍 Overview

This project demonstrates how to build a real-time image classification web application using a Convolutional Neural Network (CNN) and Flask. The user uploads an image, and the model predicts its class.

---

## 🗂️ Directory Structure

Refer to the structure provided above.

---

## 🚀 How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
2. Train the Model

python src/train.py
This will generate cnn_model.h5 inside the outputs/ directory.

3. Launch the Web App

python src/main.py
Visit http://127.0.0.1:5000 in your browser.
```
🧪 Testing
Run:

python test/test_webapp.py
To simulate image upload and get predictions.

📸 Visuals
UI	Result

✅ Features
Simple Flask web interface

Image upload support

Real-time inference

CNN model for multi-class image classification

Styled using HTML/CSS

🧠 Learning Outcomes
Train a CNN for image classification

Deploy it in a Flask web app

Handle file uploads and image processing

Use TensorFlow/Keras for model training

Use Flask + Jinja2 + Pillow + OpenCV for deployment

📌 Sample Input
Image upload (e.g., a picture of a cat 🐱)

📤 Output:

{
  "prediction": "Cat"
}
🧵 Stay Tuned
This is Day 29 of my #30DaysMLProjects series.

📅 Coming up next (Day 30):
AI Portfolio Showcase & Final Reflection!
