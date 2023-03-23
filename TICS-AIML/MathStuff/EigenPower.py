# https://numpy.org/doc/stable/reference/generated/numpy.matmul.html
import numpy as np

SIZE = 3
A = np.array([[-1, 0, 1],
              [-3, 4, 1],
              [0, 0, 2]])
# A = np.array([[5, 0],
#               [2, 1]])
x = np.array([1] * SIZE)

prevx = None
for j in range(10000):
    prevx = x
    x = np.matmul(A, x)
    m = abs(x).max()
    x = x / m
    print(x, m)
    if np.array_equiv(prevx, x):
        print("\n\n\n\n\nMatrix:")
        print(A)
        print("Estimated dominant eigenvalue:", m)
        break
print("Estimated dominant eigenvector:", [val if val > 0.001 else 0 for val in x])
