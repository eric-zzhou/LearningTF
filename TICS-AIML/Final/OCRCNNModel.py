"""
Eric Zhou
Ms. Lewellen
TICS:AIML
March 9th, 2022

This file will create the ML model made with TF and Keras that will distinguish handwritten letters and numbers. This
model will be saved and used for the OCR after some other processing.
"""
from keras.datasets import mnist
from tensorflow.python.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.utils.np_utils import normalize, to_categorical
from sklearn.model_selection import train_test_split
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import cv2

# Load digits mnist data
(train_data, train_labels), (test_data, test_labels) = mnist.load_data()
# Stacking data and labels for combining with a-z
digits_data = np.vstack([train_data, test_data])
digits_labels = np.hstack([train_labels, test_labels])

# Load letters data
az_data = []
az_labels = []
for row in open("A_Z Handwritten Data.csv"):
    row = row.split(",")
    # Label
    label = int(row[0])
    az_labels.append(label)
    # Data (image)
    image = np.array([int(x) for x in row[1:]], dtype="uint8")
    image = image.reshape((28, 28))
    az_data.append(image)

# Converting to float and int numpy arrays (to match digits data)
az_data = np.array(az_data, dtype='float32')
az_labels = np.array(az_labels, dtype="int")
# Incrementing alphabet labels since digits uses 0 - 9
az_labels += 10

# Combining the datasets
data = np.vstack([az_data, digits_data])
labels = np.hstack([az_labels, digits_labels])

# Add a dimension to the image for CNN and normalize pixels (0 to 1)
data = np.expand_dims(data, axis=-1)
data = normalize(data, axis=1)


# Shows some examples with the correct answer with the image of the number
sns.set(font_scale=2)  # font size
label_key = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
index = np.random.choice(np.arange(len(data)), 24, replace=False)  # randomly chosen 24
figure, axes = plt.subplots(nrows=4, ncols=6, figsize=(16, 9))  # 4 x 6 plot, changes size
for item in zip(axes.ravel(), data[index], labels[index]):  # plot all the new things
    axes, image, target = item
    axes.imshow(image, cmap=plt.cm.gray_r)
    axes.set_xticks([])
    axes.set_yticks([])
    axes.set_title(label_key[target.item()])
plt.tight_layout()
plt.show()

# Change labels to array thing for training
labels = to_categorical(labels)

# Train test split for data
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.1, random_state=11)

cnn = Sequential()
cnn.add(Conv2D(filters=64,
               kernel_size=(3, 3),  # 3 x 3 is pretty normal for kernel size,
               activation='relu', input_shape=(28, 28, 1)))  # convolutional layer for 2D, need input shape
cnn.add(MaxPooling2D(pool_size=(2, 2)))  # makes a pool size, 2 by 2 is normal (too large = over generalize)
cnn.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))  # relu as activation
cnn.add(MaxPooling2D(pool_size=(2, 2)))  # pool size again
cnn.add(Flatten())  # flattens the layer
cnn.add(Dense(units=256, activation='relu'))  # dense layer with 256 nodes that uses relu as activation
cnn.add(Dense(units=128, activation='relu'))  # dense layer with 128 nodes that uses relu as activation
# Output layer
cnn.add(Dense(units=36, activation='softmax'))  # dense layer with 10 nodes that uses softmax as activation
print(cnn.summary())  # print summary of model
cnn.compile(optimizer='adam',  # using adam to tune weights and optimize
            loss='categorical_crossentropy',  # using categorical_crossentropy to calculate loss
            metrics=['accuracy'])  # using accuracy as the metric for how good it does

# tensorboard_callback = TensorBoard(log_dir=f'./Logs/mnist{time.time()}',
#                                     histogram_freq=1, write_graph=True)
# Fitting and checking model. Batch size of 64 before checking and going 5 epochs/rounds
cnn.fit(X_train, y_train, epochs=2, batch_size=64, validation_split=0.1)
# epochs are big steps where you run all of your training data through. major adjustments between each epoch
# small adjustments between batches
# validation split keeps last 10% of data for checking accuracy and loss every epoch

# saving model
cnn.save("ocrcnn.h5")
# cnn = load_model('mnist_cnn.h5')

# todo: maybe do class weights
# # Account for awkward data (way more letters than numbers)
# class_totals = labels.sum(axis=0)
# class_weight = {}
# for i in range(0, len(class_totals)):
#     class_weight[i] = class_totals.max() / class_totals[i]


