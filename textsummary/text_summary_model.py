from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.layers import Input, LSTM, Embedding, Dense, Concatenate, TimeDistributed, Bidirectional, \
    Attention
from keras.models import Model
from keras import backend as K
from keras.callbacks import EarlyStopping
from textsummary.attention import AttentionLayer
import pickle

MAX_LEN_TEXT = 80
MAX_LEN_SUMMARY = 10

data = pickle.load(open('clean_data.pkl', 'rb'))

# split into training and te data
x_tr, x_val, y_tr, y_val = train_test_split(data['cleaned_text'], data['cleaned_summary'], test_size=0.1,
                                            random_state=0, shuffle=True)

# -------------------------------------------------------------------------------------------------------------------- #
# prepare a tokenizer for reviews on training data
x_tokenizer = Tokenizer()
x_tokenizer.fit_on_texts(list(x_tr))

# convert text sequences into integer sequences
x_tr = x_tokenizer.texts_to_sequences(x_tr)
x_val = x_tokenizer.texts_to_sequences(x_val)

# padding zero upto maximum length
x_tr = pad_sequences(x_tr, maxlen=MAX_LEN_TEXT, padding='post')
x_val = pad_sequences(x_val, maxlen=MAX_LEN_TEXT, padding='post')

x_voc_size = len(x_tokenizer.word_index) + 1

# -------------------------------------------------------------------------------------------------------------------- #
# preparing a tokenizer for summary on training data
y_tokenizer = Tokenizer()
y_tokenizer.fit_on_texts(list(y_tr))  # makes a list to track how many times each letter appears

# convert summary sequences into integer sequences
y_tr = y_tokenizer.texts_to_sequences(y_tr)
y_val = y_tokenizer.texts_to_sequences(y_val)

# padding zero up to maximum length
y_tr = pad_sequences(y_tr, maxlen=MAX_LEN_SUMMARY, padding='post')
y_val = pad_sequences(y_val, maxlen=MAX_LEN_SUMMARY, padding='post')

y_voc_size = len(y_tokenizer.word_index) + 1

# -------------------------------------------------------------------------------------------------------------------- #
K.clear_session()
latent_dim = 500

# Encoder
encoder_inputs = Input(shape=(MAX_LEN_TEXT,))
enc_emb = Embedding(x_voc_size, latent_dim, trainable=True)(encoder_inputs)

# LSTM 1
encoder_lstm1 = LSTM(latent_dim, return_sequences=True, return_state=True)
encoder_output1, state_h1, state_c1 = encoder_lstm1(enc_emb)

# LSTM 2
encoder_lstm2 = LSTM(latent_dim, return_sequences=True, return_state=True)
encoder_output2, state_h2, state_c2 = encoder_lstm2(encoder_output1)

# LSTM 3
encoder_lstm3 = LSTM(latent_dim, return_state=True, return_sequences=True)
encoder_outputs, state_h, state_c = encoder_lstm3(encoder_output2)

# -------------------------------------------------------------------------------------------------------------------- #
# Set up the decoder.
decoder_inputs = Input(shape=(None,))
dec_emb_layer = Embedding(y_voc_size, latent_dim, trainable=True)
dec_emb = dec_emb_layer(decoder_inputs)

# LSTM using encoder_states as initial state
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, decoder_fwd_state, decoder_back_state = decoder_lstm(dec_emb, initial_state=[state_h, state_c])

# Attention Layer
attn_layer = AttentionLayer(name='attention_layer')
attn_out, attn_states = attn_layer([encoder_outputs, decoder_outputs])

# Concat attention output and decoder LSTM output
decoder_concat_input = Concatenate(axis=-1, name='concat_layer')([decoder_outputs, attn_out])

# Dense layer
decoder_dense = TimeDistributed(Dense(y_voc_size, activation='softmax'))
decoder_outputs = decoder_dense(decoder_concat_input)

# -------------------------------------------------------------------------------------------------------------------- #
# Define the model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.summary()

# actual model
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')  # loss function
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)  # stops when loss increases

# -------------------------------------------------------------------------------------------------------------------- #
# training
history = model.fit([x_tr, y_tr[:, :-1]], y_tr.reshape(y_tr.shape[0], y_tr.shape[1], 1)[:, 1:], epochs=50,
                    callbacks=[es], batch_size=512,
                    validation_data=([x_val, y_val[:, :-1]], y_val.reshape(y_val.shape[0], y_val.shape[1], 1)[:, 1:]))

# diagnostic plot
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()
