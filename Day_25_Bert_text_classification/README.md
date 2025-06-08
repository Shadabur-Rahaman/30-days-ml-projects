📘 Day 25 – BERT IMDB Sentiment Classification
Fine-tuning a pretrained BERT model on the IMDB movie reviews dataset to classify reviews as positive or negative.

🧠 Objective
Use HuggingFace Transformers to:

Tokenize and encode IMDB reviews

Fine-tune BERT for binary classification

Save model artifacts and tokenizer configs

Evaluate with confusion matrix, accuracy, and loss plots

🗂️ Directory Structure

Day25_BERT_IMDB_Sentiment/
├── bert_imdb_sentiment/                      # Tokenizer and vocab files
│   ├── config.json
│   ├── special_tokens_map.json
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── vocab.txt
├── images/
│   ├── training_loss_accuracy.png            # Training loss/accuracy curves
│   └── confusion_matrix.png                  # Confusion matrix from test results
├── notebooks/
│   └── Day_25_Bert_text_classification.ipynb # Main notebook
├── src/
│   ├── dataset.py                            # Dataset loading and preprocessing
│   ├── model.py                              # BERTClassifier model
│   ├── train.py                              # Training loop
│   └── utils.py                              # Plotting, saving, etc.
├── requirements.txt                          # Python dependencies
├── .gitignore                                # Git ignore patterns
└── README.md                                 # This file
🔧 Installation
Clone this repo:

bash
Copy
Edit
git clone https://github.com/your-username/Day25_BERT_IMDB_Sentiment.git
cd Day25_BERT_IMDB_Sentiment
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
🚀 Usage
You can either run the full pipeline from the notebook:

bash
Copy
Edit
notebooks/Day_25_Bert_text_classification.ipynb
Or use the modular pipeline via src/ scripts:

✅ Example:
python
Copy
Edit
from src.dataset import get_dataloaders
from src.model import BERTClassifier
from src.train import train_model
from src.utils import plot_loss_accuracy, save_model
📊 Visual Outputs
Training & Accuracy	Confusion Matrix


📚 Learnings
HuggingFace AutoTokenizer & AutoModelForSequenceClassification

BERT tokenizer vocabulary saving: tokenizer.json, vocab.txt, config.json

Training with custom DataLoaders and optimizer setup

Saving and reloading models with torch.save

Model evaluation using classification metrics (precision, recall, F1)

