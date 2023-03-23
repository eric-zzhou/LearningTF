from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

data = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = data.load_data()

# print(train_images[0])
train_images = keras.utils.normalize(train_images, axis=1)
test_images = keras.utils.normalize(test_images, axis=1)
# print(train_images[0])

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(144, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(train_images, train_labels, epochs=6)
test_loss, test_acc = model.evaluate(test_images, test_labels)
print("Tested Accuracy: ", test_acc)

model.save("mnist_model.h5")

prediction = model.predict(test_images)
for i in range(len(test_labels)):
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel("Actual: " + str(test_labels[i]))
    plt.title("Prediction: " + str(np.argmax(prediction[i])))
    print(prediction[i])
    plt.show()
