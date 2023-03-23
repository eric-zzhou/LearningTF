# katelin lewellen
# Jan 28 2022
from statistics import linear_regression
from sklearn import linear_model
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

california = fetch_california_housing()
print(california.DESCR)
print(california.data.shape)
print(california.feature_names)

# Explore with pandas
# create a dataframe
california_df = pd.DataFrame(california.data,
                             columns=california.feature_names)
# add the labels of median housing values
california_df['MedHouseValue'] = pd.Series(california.target)

# print the top few
print(california_df.head())

# print some summary
print(california_df.describe())

# Graph with seaborn
# get a sample to graph
sample_df = california_df.sample(frac=0.1, random_state=17)
sns.set(font_scale=2)
sns.set_style('whitegrid')

# put each feature into a graph
# for feature in california.feature_names:
#     plt.figure(figsize=(16, 9))
#     sns.scatterplot(data=sample_df,
#                     x=feature,
#                     y='MedHouseValue',
#                     hue='MedHouseValue',
#                     palette='cool',
#                     legend=False)
#     plt.show()

X_train, X_test, y_train, y_test = train_test_split(california.data, california.target, random_state=11)

lr = LinearRegression()
lr.fit(X=X_train, y=y_train)
for i, name in enumerate(california.feature_names):
    print(f'{name:>10}: {lr.coef_[i]}')

predicted = lr.predict(X_test)
expected = y_test
