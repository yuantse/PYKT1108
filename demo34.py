# dame as pykt0511 demo30
import matplotlib.pyplot as plt
import numpy as np

from sklearn.cluster import KMeans
# 隨機產生3群 每群500個點
X = np.r_[np.random.randn(500, 2) + [2, 2],
          np.random.randn(500, 2) + [0, -2],
          np.random.randn(500, 2) + [-2, 2]]

# 利用kmeans將資料分成k群 嘗試1次 n_init 越大越精確
k = 3
kmean = KMeans(n_init=1, n_clusters=3)
# kmeans為非監督式學習 不須先給結果y
kmean.fit(X)

print(kmean.cluster_centers_)
print(kmean.inertia_)
colors = ['c', 'm', 'y', 'k']
markers = ['o', 's', '*', '^']
# 分群塗上顏色
for i in range(k):
    dataX = X[kmean.labels_ == i]
    plt.scatter(dataX[:, 0], dataX[:, 1], c=colors[i], marker=markers[i])
    print(dataX.size)
# 畫上kmeans推論的星星
plt.scatter(kmean.cluster_centers_[:, 0], kmean.cluster_centers_[:, 1],
            marker='*', s=200, c='#C0FFEE')
plt.show()
