from sklearn.datasets import load_digits
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

digits = load_digits()
print(digits.DESCR)
# Set up the clustering estimator
kmeans = KMeans(n_clusters=10, random_state=11)
kmeans.fit(digits.data)
# algorithm = auto, copy_x=Ture, init='kmeans++', max_iter=300
# n_clusters=3, n_init=10, n_jobs=None, precompute_distances='auto',
# random_state=11, to1=0.0001, verbose=0
# Because we didn't shuffle the data, we know that the targets go from 0 - 9 over and over
print("kmean:", kmeans.labels_[:10])
print("actual:", digits.target[:10])
# print(kmeans.labels_[10:20])

# Try a different estimator to do dimensional reduction
tsne = TSNE(n_components=2, random_state=11)
digits_tsne = tsne.fit_transform(digits.data)
print(digits_tsne.shape)
# Now let's visualize it
digits_tsne_df = pd.DataFrame(digits_tsne, columns=['Component1', 'Component2'])

# plot the data on 2 dimensions
axes = sns.scatterplot(data=digits_tsne_df, x='Component1', y='Component2',
                       hue=digits.target, legend='brief', palette='cool')

# turn the centroids to two dimensions (I'm pretty sure the centroids are off but I don't know why
digits_center = tsne.fit_transform(kmeans.cluster_centers_)
dots = plt.scatter(digits_center[:, 0], digits_center[:, 1], s=100, c='k')
plt.show()
