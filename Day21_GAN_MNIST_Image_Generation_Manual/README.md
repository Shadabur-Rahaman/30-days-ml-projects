# Day 20: Chatbot with Seq2Seq Model

This project demonstrates how to build a simple chatbot using a sequence-to-sequence (Seq2Seq) model built on LSTM/GRU. The chatbot is trained on paired conversational data to learn how to respond to inputs contextually.

## 🔍 Project Highlights
- Used encoder-decoder architecture (Seq2Seq)
- Preprocessed conversation pairs for training
- Implemented training and inference using TensorFlow/Keras
- Saved responses to custom inputs

## 📁 Directory Structure
```
Day_20_Chatbot_Seq2Seq_Model/
├── notebooks/
│   └── Day_20_Chatbot_Seq2Seq_Model.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── model_seq2seq.py
│   ├── inference.py
├── outputs/
│   └── image_at_epoch_0000
    └── image_at_epoch_0010
    └── image_at_epoch_0020
    └── image_at_epoch_0030
    └── image_at_epoch_0040
    └── image_at_epoch_0050
    └── image_at_epoch_0060
    └── image_at_epoch_0070
    └── image_at_epoch_0080
    └── image_at_epoch_0090
    └── image_at_epoch_0100
├── requirements.txt
├── .gitignore
├── README.md
```

## ⚙️ How to Run
1. Install dependencies:
```bash
pip install -r requirements.txt
```
2. Run the Jupyter notebook or execute preprocessing and training scripts from `src/`
3. Use the `inference.py` script to chat with the bot interactively.

## 🧠 Technologies Used
- Python, TensorFlow, Keras
- NLTK for text cleaning
- Encoder-Decoder LSTM/GRU models

---
