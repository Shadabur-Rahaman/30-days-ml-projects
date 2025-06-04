import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences

def build_inference_models(model, latent_dim, vocab_size, tokenizer, max_len):
    # Encoder
    encoder_inputs = model.input[0]
    encoder_emb = model.layers[2]
    encoder_lstm = model.layers[3]
    encoder_outputs, state_h_enc, state_c_enc = encoder_lstm(encoder_emb.output)
    encoder_model = Model(encoder_inputs, [state_h_enc, state_c_enc])

    # Decoder
    decoder_inputs = model.input[1]
    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

    decoder_emb_layer = model.layers[4]
    decoder_lstm = model.layers[5]
    decoder_dense = model.layers[6]

    dec_emb = decoder_emb_layer(decoder_inputs)
    decoder_outputs, state_h, state_c = decoder_lstm(dec_emb, initial_state=decoder_states_inputs)
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + [state_h, state_c]
    )

    return encoder_model, decoder_model

def decode_sequence(input_seq, encoder_model, decoder_model, tokenizer, max_len):
    reverse_word_index = {idx: word for word, idx in tokenizer.word_index.items()}
    reverse_word_index[0] = ''

    states_value = encoder_model.predict(input_seq)

    target_seq = np.zeros((1, max_len))
    target_seq[0, 0] = tokenizer.word_index['start']

    stop_condition = False
    decoded_sentence = ''

    for i in range(max_len):
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

        sampled_token_index = np.argmax(output_tokens[0, i, :])
        sampled_word = reverse_word_index.get(sampled_token_index, '')

        if sampled_word == 'end' or sampled_word == '':
            break

        decoded_sentence += ' ' + sampled_word

        target_seq[0, i] = sampled_token_index
        states_value = [h, c]

    return decoded_sentence.strip()
