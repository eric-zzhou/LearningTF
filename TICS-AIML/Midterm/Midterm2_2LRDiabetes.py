from sklearn import linear_model
from sklearn.datasets import load_diabetes
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

disease = load_diabetes()

# Reduce the features on the data so we can plot in 2d
tsne = TSNE(n_components=2, random_state=11)
reduced_data = tsne.fit_transform(disease.data)

# Plot in colors by progression of the disease
dots = plt.scatter(reduced_data[:, 0], reduced_data[:, 1],
                   c=disease.target, cmap=plt.cm.get_cmap('nipy_spectral_r', 10))
colorbar = plt.colorbar(dots)

plt.show()

# MY CODE:
# splitting data into training and testing
X_train, X_test, y_train, y_test = train_test_split(disease.data, disease.target, random_state=11)

# actual linear regression
lr = LinearRegression()
lr.fit(X=X_train, y=y_train)

# looking at coefficients for each attribute
for i, name in enumerate(disease.feature_names):
    print(f'{name:>10}: {lr.coef_[i]}')

# predictions and true values for visualization
predicted = lr.predict(X_test)
expected = y_test

# we will visualize it
# set up the data in the way pandas likes it
df = pd.DataFrame()
df['Expected'] = pd.Series(expected)
df['Predicted'] = pd.Series(predicted)
figure = plt.figure(figsize=(9, 9))

axes = sns.scatterplot(data=df, x='Expected', y='Predicted', hue='Predicted', palette='cool', legend=False)

# Set up axes
start = min(expected.min(), predicted.min())
end = max(expected.max(), predicted.max())
axes.set_xlim(start, end)
axes.set_ylim(start, end)

# add a line where perfect prediction would be
line = plt.plot([start, end], [start, end], 'k--')
plt.show()
