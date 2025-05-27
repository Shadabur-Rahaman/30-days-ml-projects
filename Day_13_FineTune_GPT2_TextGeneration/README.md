# 🤖 Day 13 – Fine-Tuning GPT-2 for Text Generation

In this project, I fine-tuned OpenAI’s GPT-2 model on a custom dataset using the HuggingFace Transformers library. The fine-tuned model can generate coherent and stylistically consistent text based on the given corpus.

![GPT-2 Text Generation](images/gpt2_textgen_sample.png)

## 🔍 Overview

- Model: GPT-2 (124M parameters)
- Dataset: Custom text corpus
- Library: HuggingFace Transformers
- Tasks: Text preprocessing, tokenization, fine-tuning, generation
```bash
## 📁 File Structure

Day13_FineTune_GPT2_TextGeneration/
├── data/
│ └── custom_corpus.txt
├── notebooks/
│ └── Day13_FineTune_GPT2_TextGeneration.ipynb
├── outputs/
│ ├── generated_text_samples.txt
│ └── model_checkpoints/
├── src/
│ └── gpt2_finetune_utils.py
├── requirements.txt
├── .gitignore
└── README.md


## ⚙️ How to Run

pip install -r requirements.txt
Then open and run:


notebooks/Day13_FineTune_GPT2_TextGeneration.ipynb
Make sure your custom_corpus.txt is placed under data/.

📚 Learning Outcomes
Understand how transformers learn language

Apply transfer learning using GPT-2

Fine-tune a pre-trained language model on custom data

Generate creative, contextual text outputs

