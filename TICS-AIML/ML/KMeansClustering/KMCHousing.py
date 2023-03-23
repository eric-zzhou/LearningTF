

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

california = fetch_california_housing()
print(california.DESCR)

s_data = list(california.data)[0::6]
s_price = list(california.target)[0::6]

# Cut out information I think is not important using panda dataframe
california_df = pd.DataFrame(s_data, columns=california.feature_names)
california_df = california_df.drop(['Population', 'AveOccup'], axis=1)
print(california_df)
s_data = california_df.values.tolist()
print(s_data[0])

# find the best k value between 2 to 50 (2 because 1 wouldn't make sense and 50 is just randomly chosen)
besterror = 1000000000000000
bestk = -1
for i in range(2, 50):
    kmeans = KMeans(n_clusters=i, random_state=11)
    kmeans.fit(s_data)
    # alogorithm = auto, copy_x=Ture, init='kmeans++', max_iter=300
    # n_clusers=3, n_init=10, n_jobs=None, precompute_distances='auto',
    # random_state=11, to1=0.0001, verbose=0
    # Because we didn't shuffle the data, we know that
    # The first 50 are in one cluster, then the next 50, then the last 50
    sq_error = 0
    for j in range(50):
        sq_error += (kmeans.labels_[j] - s_price[j])**2
    if sq_error < besterror:
        besterror = sq_error
        bestk = i
    print(f"i: {i}, squared error: {sq_error}")
    print("kmean:", kmeans.labels_[:50])
    print("actual:", s_price[:50], "\n")
print(f"besterror: {besterror}, bestk: {bestk}")

# Set up the clustering estimator
kmeans = KMeans(n_clusters=bestk, random_state=11)
kmeans.fit(s_data)
print("kmean:", kmeans.labels_[:50])
print("actual:", s_price[:50], "\n")

# Try a different estimator to do dimensional reduction
pca = PCA(n_components=2, random_state=11)
pca.fit(s_data)
california_pca = pca.transform(s_data)
print(california_pca.shape)
# Now let's visualize it
iris_pca_df = pd.DataFrame(california_pca, columns=['Component1', 'Component2'])
iris_pca_df['prices'] = s_price
# plot the data on 2 dimensions
axes = sns.scatterplot(data=iris_pca_df, x='Component1', y='Component2',
                       hue='prices', legend='brief', palette='cool')
# turn the centroids to two dimensions
california_center = pca.transform(kmeans.cluster_centers_)
dots = plt.scatter(california_center[:, 0], california_center[:, 1], s=100, c='k')
plt.show()
