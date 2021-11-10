# pykt0511 demo31
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans

# 隨機產生3群 每群5000個點
Q = 5000
X = np.r_[np.random.randn(Q, 2) + [2, 2],
          np.random.randn(Q, 2) + [0, -2],
          np.random.randn(Q, 2) + [-2, 2]]

# 嘗試不同的群數
inertias = []
for k in range(1, 10):
    kmean = KMeans(n_init=5, n_clusters=k)
    kmean.fit(X)
    inertias.append(kmean.inertia_)

# 從inertias觀察 當分群數為3時有顯著的效果
plt.plot(range(1, 10), inertias)
plt.show()
