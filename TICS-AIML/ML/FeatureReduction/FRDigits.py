from sklearn.datasets import load_digits
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

digits = load_digits()
# print(digits.DESCR)
# Reduce the features on the data so we can plot in 2d
tsne = TSNE(n_components=2, random_state=11)
reduced_data = tsne.fit_transform(digits.data)

# Plot 1 all black
# dots = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c='black')

# Plot 2 in colors for digits
dots = plt.scatter(reduced_data[:, 0], reduced_data[:, 1],
                   c=digits.target, cmap=plt.cm.get_cmap('nipy_spectral_r', 10))

colorbar = plt.colorbar(dots)
plt.show()
