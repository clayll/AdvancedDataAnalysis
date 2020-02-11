import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression

# 构造一些数据点
centers = [[-5, 0], [0, 1.5], [5, -1]]
X, y = make_blobs(n_samples=1000, centers=centers, random_state=40)
transformation = [[0.4, 0.2], [-0.4, 1.2]]
X = np.dot(X, transformation)
plt.scatter(X[:,0],X[:,1],c=y)
plt.show()

clf = LogisticRegression(solver='sag', max_iter=100, random_state=42).fit(X, y)

print(clf.coef_)

print(clf.intercept_)

