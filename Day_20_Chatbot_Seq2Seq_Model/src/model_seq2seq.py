import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense

def build_seq2seq_model(vocab_size, embedding_dim=256, units=512, max_len=20):
    encoder_inputs = Input(shape=(max_len,))
    embed = Embedding(vocab_size, embedding_dim)(encoder_inputs)
    encoder_lstm, state_h, state_c = LSTM(units, return_state=True)(embed)
    encoder_states = [state_h, state_c]

    decoder_inputs = Input(shape=(max_len,))
    decoder_embed = Embedding(vocab_size, embedding_dim)(decoder_inputs)
    decoder_lstm = LSTM(units, return_sequences=True, return_state=False)
    decoder_outputs = decoder_lstm(decoder_embed, initial_state=encoder_states)
    decoder_dense = Dense(vocab_size, activation='softmax')
    output = decoder_dense(decoder_outputs)

    model = Model([encoder_inputs, decoder_inputs], output)
    return model
