import re
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def load_conversations(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        lines = file.read().split('\n')
    questions, answers = [], []
    for line in lines:
        if '\t' in line:
            q, a = line.split('\t')
            questions.append(clean_text(q))
            answers.append(clean_text(a))
    return questions, answers

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9]+", " ", text)
    return text.strip()

def tokenize_and_pad(questions, answers, vocab_size=10000, max_len=20):
    tokenizer = Tokenizer(num_words=vocab_size, filters='')
    tokenizer.fit_on_texts(questions + answers)

    questions_seq = tokenizer.texts_to_sequences(questions)
    answers_seq = tokenizer.texts_to_sequences(answers)

    questions_padded = pad_sequences(questions_seq, maxlen=max_len, padding='post')
    answers_padded = pad_sequences(answers_seq, maxlen=max_len, padding='post')

    return questions_padded, answers_padded, tokenizer
