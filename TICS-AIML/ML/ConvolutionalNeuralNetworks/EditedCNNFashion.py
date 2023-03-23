# Eric Zhou
# EPS TICS:AIML
# February 11th, 2022

# A file creating a multilayer CNN to process the fashion_mnist dataset and recognize clothes

from keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load dataset into training and testing and looking at shape (length / size of arrays)
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
X_train, X_discard, y_train, y_discard = train_test_split(X_train, y_train, random_state=11, test_size=0.75)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# Shows some examples with the correct answer with the image of the number
# sns.set(font_scale=2)  # font size
# index = np.random.choice(np.arange(len(X_train)), 24, replace=False)  # randomly chosen
# figure, axes = plt.subplots(nrows=4, ncols=6, figsize=(16, 9))  # 4 x 6 plot, changes size
# for item in zip(axes.ravel(), X_train[index], y_train[index]):  # plot all the new things
#     axes, image, target = item
#     axes.imshow(image, cmap=plt.cm.gray_r)
#     axes.set_xticks([])
#     axes.set_yticks([])
#     axes.set_title(target)
# plt.tight_layout()
# plt.show()


# Data preprocessing
# Add a new dimension to the tensor
X_train = X_train.reshape((15000, 28, 28, 1))
X_test = X_test.reshape((10000, 28, 28, 1))

# Normalize the data and change everything to 1 after making it a float
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Sequentially creating neural network
cnn = Sequential()
cnn.add(Conv2D(filters=64,
               kernel_size=(3, 3),  # 3 x 3 is pretty normal for kernel size,
               activation='relu', input_shape=(28, 28, 1)))  # convolutional layer for 2D, need input shape
cnn.add(MaxPooling2D(pool_size=(2, 2)))  # makes a pool size, 2 by 2 is normal (too large = over generalize)
cnn.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))  # relu as activation
cnn.add(MaxPooling2D(pool_size=(2, 2)))  # pool size again
cnn.add(Flatten())  # flattens the layer
cnn.add(Dense(units=128, activation='relu'))  # dense layer with 128 nodes that uses relu as activation
# Output layer
cnn.add(Dense(units=10, activation='softmax'))  # dense layer with 10 nodes that uses softmax as activation
print(cnn.summary())  # print summary of model
cnn.compile(optimizer='adam',  # using adam to tune weights and optimize
            loss='categorical_crossentropy',  # using categorical_crossentropy to calculate loss
            metrics=['accuracy'])  # using accuracy as the metric for how good it does

# tensorboard_callback = TensorBoard(log_dir=f'./Logs/mnist{time.time()}',
#                                     histogram_freq=1, write_graph=True)
# Fitting and checking model. Batch size of 64 before checking and going 5 epochs/rounds
cnn.fit(X_train, y_train, epochs=3, batch_size=64, validation_split=0.1)
# epochs are big steps where you run all of your training data through. major adjustments between each epoch
# small adjustments between batches
# validation split keeps last 10% of data for checking accuracy and loss every epoch

# saving model
cnn.save("fashion_mnist_cnn.h5")
# cnn = load_model('mnist_cnn.h5')

# Checking loss and accuracy on test data
loss, accuracy = cnn.evaluate(X_test, y_test)
print(loss)
print(accuracy)

# Using predict to test on testing data
predictions = cnn.predict(X_test)
print(y_test[0])

images = X_test.reshape((10000, 28, 28))
incorrect_predictions = []

for i, (p, e) in enumerate(zip(predictions, y_test)):
    predicted, expected = np.argmax(p), np.argmax(e)
    if predicted != expected:
        incorrect_predictions.append((i, images[i], predicted, expected))
print("number of incorrect: ", len(incorrect_predictions))