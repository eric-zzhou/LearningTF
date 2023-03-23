from sklearn.datasets import fetch_california_housing
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

california = fetch_california_housing()
print(california.DESCR)

# Take subset of data since it's too large
s_data = list(california.data)[0::4]
s_price = list(california.target)[0::4]

# Reduce the features on the data so we can plot in 2d
tsne = TSNE(n_components=2, random_state=11)
reduced_data = tsne.fit_transform(s_data)

# Plot 1 all black
# dots = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c='black')

# Plot 2 in colors for digits
dots = plt.scatter(reduced_data[:, 0], reduced_data[:, 1],
                   c=s_price, cmap=plt.cm.get_cmap('nipy_spectral_r', 10))
colorbar = plt.colorbar(dots)
plt.show()