# Eric Zhou
# 1/26/2022
# EPS TiCS:AIML Class 14

# An example of K nearest neighbors classification on a digit dataset

# import dataset
from sklearn.datasets import load_digits
# from sklearn.datasets import load_iris for hw

# classification imports
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report

# pretty visualization stuff
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# load dataset
digits = load_digits()

# prints shape of dataset: rows (data) and columns (attributes)
# print(digits.DESCR)  # description
print(digits.data.shape)
# split the dataset
# X values are samples (images)
# Y values are labels or targets (answer)
# Train dataset
# Test dataset
# splits 75% for train and 25% for test
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, random_state=11)

# Sets up the model and trains it
knn = KNeighborsClassifier()
knn.fit(X=X_train, y=y_train)

# See how well it performs on test dataset
expected = y_test
predicted = knn.predict(X=X_test)

# print first 20 expected and predicted
print("expected", expected[:20])
print("predicted", predicted[:20])

# list of all wrong predictions
wrong = [(p, e) for (p, e) in zip(predicted, expected) if p != e]
print("wrong", wrong)

# accuracy score rounded to 2 decimal places percent
print(f'{knn.score(X_test, y_test):.2%}')

# Visualize
figure, axes = plt.subplots(nrows=4, ncols=6, figsize=(6, 4))  # 4 x 6 subplots
for item in zip(axes.ravel(), digits.images, digits.target):  # zips together images and targets
    axes, image, target = item  # getting parts of iem
    axes.imshow(image, cmap=plt.cm.gray_r)  # shows image in grayscale
    axes.set_xticks([])  # removes ticks of the axis
    axes.set_yticks([])  # removes ticks of the axis
    axes.set_title(target)  # setting title of subplot to target
plt.tight_layout()  # layout
plt.show()  # show graph


