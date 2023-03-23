from sklearn.datasets import load_iris
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

iris = load_iris()
print(iris.DESCR)

# Set up a pandas data frame and plot it pairwise
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['species'] = [iris.target_names[i] for i in iris.target]
# sns.set(font_scale=1.1)
# sns.set_style('whitegrid')
# gid = sns.pairplot(data=iris_df, vars=iris_df.columns[0:4], hue='species')
# plt.show()
# Set up the clustering estimator
kmeans = KMeans(n_clusters=3, random_state=11)
kmeans.fit(iris.data)
# alogorithm = auto, copy_x=Ture, init='kmeans++', max_iter=300
# n_clusers=3, n_init=10, n_jobs=None, precompute_distances='auto',
# random_state=11, to1=0.0001, verbose=0
# Because we didn't shuffle the data, we know that
# The first 50 are in one cluster, then the next 50, then the last 50
print(kmeans.labels_[0:50])
print(kmeans.labels_[50:100])
print(kmeans.labels_[100:150])

# # Now try it with just two clusters
# # Set up the clustering estimator
# kmeans = KMeans(n_clusters=2, random_state=11)
# kmeans.fit(iris.data)
# print(kmeans.labels_[0:50])
# print(kmeans.labels_[50:150])

# Try a different estimator to do dimensional reduction
pca = PCA(n_components=2, random_state=11)
pca.fit(iris.data)
iris_pca = pca.transform(iris.data)
print(iris_pca.shape)
# Now let's visualize it
iris_pca_df = pd.DataFrame(iris_pca, columns=['Component1', 'Component2'])
iris_pca_df['species'] = iris_df.species
# plot the data on 2 dimensions
axes = sns.scatterplot(data=iris_pca_df, x='Component1', y='Component2',
                       hue='species', legend='brief', palette='cool')
# turn the centroids to two dimensions
iris_center = pca.transform(kmeans.cluster_centers_)
dots = plt.scatter(iris_center[:, 0], iris_center[:, 1], s=100, c='k')
plt.show()
