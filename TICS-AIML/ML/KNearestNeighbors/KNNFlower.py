# Eric Zhou
# 1/26/2022
# EPS TiCS:AIML Class 14

# An example of K nearest neighbors classification on a flower iris dataset

'''
1. Dataset is a set of 150 data values with a target of type of flower (out of setosa, versicolor, and virginica). The
   attributes included in the dataset are sepal length, sepal width, petal length, and petal width.
   As for K-nearest neighbors, it basically graphs each datapoint in a multidimensional plane (number of dimensions
   is equal to number of attributes) with a lot of already defined data points where you know the correct corresponding
   target. The K value defines how many of closest neighbors do you consider when categorizing your data point (based
   on simple majority). You are trying to find the optimal (or close to optimal) K value to categorize data points and
   testing the accuracy of your model.
2. Sepal length: 5.7 cm, sepal width: 4.4 cm, petal length: 1.5 cm, and petal width: 0.4 cm
3. 97.37% accurate
4. 1 wrong answer
'''

# import dataset
from sklearn.datasets import load_iris

# classification imports
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report

# pretty visualization stuff
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# load dataset
flowers = load_iris()
flower_names = list(flowers.target_names)

# prints shape of dataset: rows (data) and columns (attributes)
# print(digits.DESCR)  # description
print("shape:", flowers.data.shape)

# finding values of 16th number (15 because starts at 0)
print(flowers.data[15])

# split the dataset
# X values are samples (images)
# Y values are labels or targets (answer)
# Train dataset
# Test dataset
# splits 75% for train and 25% for test
X_train, X_test, y_train, y_test = train_test_split(flowers.data, flowers.target, random_state=22)

# Sets up the model and trains it
knn = KNeighborsClassifier()
knn.fit(X=X_train, y=y_train)

# See how well it performs on test dataset
expected = y_test
predicted = knn.predict(X=X_test)

# print first 20 expected and predicted
print("expected:", expected[:20])
print("predicted:", predicted[:20])

# list of all wrong predictions
wrong = [(p, e) for (p, e) in zip(predicted, expected) if p != e]
print("wrong:", wrong)

# accuracy score (formatted) rounded to 2 decimal places percent
print(f'{knn.score(X_test, y_test):.2%}')

# confusion matrix: a table showing all predicted and expected answers to see the categorization in numbers
confusion = confusion_matrix(y_true=expected, y_pred=predicted)

# visualize the confusion matrix using matplotlib
confusion_df = pd.DataFrame(confusion, index=range(3), columns=range(3))
axes = sns.heatmap(confusion_df, annot=True, cmap='nipy_spectral_r')
axes.set_title("KNNFlower Confusion Matrix")
plt.show()

max_val = 0.0
maxind = 0
for k in range(1, 60, 1):
    kfold = KFold(n_splits=10, random_state=22, shuffle=True)
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(estimator=knn, X=flowers.data, y=flowers.target, cv=kfold)
    if scores.mean() > max_val:
        max_val = scores.mean()
        maxind = k
    print(f'k={k:<2}; mean accuracy={scores.mean():.2%}; ' +
          f'standard deviation={scores.std():.2%}')

print(f'Best L value: {maxind}, Accuracy: {max_val}')
