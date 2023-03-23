# Katelin Lewellen
# EPS TICS/AIML Day 20
# Feb 11 2022 
#
# A demo of a neural network performing sentiment analysis on IMDB


from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding
from sklearn.model_selection import train_test_split

# The dataset includes a LOT of words 88,000 unique ones
# but thats way too much for our machine to process
# If we limit the number of words to 10,000, we can reasonably 
# expect it to complete locally
# This limit only brings in the 10k *most frequent* words from IMDB
number_of_words = 10000

# load in the dataset and split it into test and train
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=number_of_words)

###################### SECTION : EXPLORING DATA ################################
# find out the characteristics
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# print out a random sample of what the data looks like
# probably an array of numbers, each indicating a word
print(X_train[123])

# This is a label of positive or negative (1 or 0)
print(y_train[123])

# the word encodings don't use the values 0, 1, or 2
# 0 is reserved for padding to get the same length for all reviews
# 1 is reserved for a start of a text sequence
# 2 is reserved for unknown words
# Let's write some functions to actually translate num<->word
word_to_index = imdb.get_word_index()
index_to_word = {index: word for (word, index) in word_to_index.items()}

# What number is "great" encoded as
print(word_to_index['great'])

# what are the top 50 words to look at?
top50 = [index_to_word[i] for i in range(1, 51)]
print(top50)

# Let's decode just review 123
rev123 = ' '.join([index_to_word.get(i - 3, '?') for i in X_train[123]])
print(rev123)

###################### SECTION : PREPARING DATA ################################ 
# We need to get all reviews to be the same length, they are not currently
# this function will allow us to reshape the data by adding 0s (padding) to end
words_per_review = 200
X_train = pad_sequences(X_train, maxlen=words_per_review)
X_test = pad_sequences(X_test, maxlen=words_per_review)

print(X_train.shape)
print(X_test.shape)

# Now let's split data for our validation run after each epoch
# for Covnets, we just took the last 10% for validation, but let's actually
# take a random one for this dataset
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test,
                                                random_state=11, test_size=0.20)

print(X_test.shape)
print(X_val.shape)

###################### SECTION : CREATING NEURAL NETWORK ####################### 
rnn = Sequential()
rnn.add(Embedding(input_dim=number_of_words, output_dim=128,
                  input_length=words_per_review))

# units: start with nodes between the number of classes (2 - pos/neg) and
#   the length of the input sequence (200)
# dropout: the percentage of neurons to randomly disable when processing in/out
#   like pooling, reduces overfitting
# recurrent_dropuout: the percentage of neurons to randomly disable when the 
#   layer is fed back into the layer again. 
rnn.add(LSTM(units=128, dropout=0.2, recurrent_dropout=0.2))

# sigmoid works well for binary classification
rnn.add(Dense(units=1, activation='sigmoid'))

rnn.compile(optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy'])

print(rnn.summary())

###################### SECTION : TRAINING NEURAL NETWORK ####################### 

rnn.fit(X_train, y_train, epochs=10, batch_size=32,
        validation_data=(X_val, y_val))

results = rnn.evaluate(X_test, y_test)

