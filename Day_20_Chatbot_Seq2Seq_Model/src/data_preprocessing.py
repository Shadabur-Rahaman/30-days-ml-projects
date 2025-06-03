import nltk
import re
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9?.!,¿]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text

def tokenize_data(questions, answers, vocab_size=5000, max_len=20):
    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(questions + answers)
    
    question_seq = tokenizer.texts_to_sequences(questions)
    answer_seq = tokenizer.texts_to_sequences(answers)
    
    question_padded = pad_sequences(question_seq, maxlen=max_len, padding='post')
    answer_padded = pad_sequences(answer_seq, maxlen=max_len, padding='post')
    
    return tokenizer, question_padded, answer_padded
