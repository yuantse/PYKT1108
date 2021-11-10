# same as pykt0511 demo28

from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np

# 隨機產生3組 每組50個的點 用roll串接
X = np.r_[np.random.randn(50, 2) + [2, 2],
          np.random.randn(50, 2) + [0, -2],
          np.random.randn(50, 2) + [-2, 2]]

# 畫圖
print(X.shape)
[plt.scatter(e[0], e[1], c='black', s=7) for e in X]

# 隨機產生3個點
k = 3
# 隨機產生3個x
C_x = np.random.uniform(np.min(X[:, 0]), np.max(X[:, 0]), size=k)
print(C_x)
# 隨機產生3個y
C_y = np.random.uniform(np.min(X[:, 1]), np.max(X[:, 1]), size=k)
print(C_y)
# 組合x, y成為3組座標(3個點)
C = np.array(list(zip(C_x, C_y)), dtype=np.float32)
print(C)
# 畫圖
plt.scatter(C_x, C_y, marker='*', s=200, c='#C0FFEE')
plt.show()

# 手動執行kmean
def dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)


C_old = np.zeros(C.shape)
clusters = np.zeros(len(X))
delta = dist(C, C_old, None)
print(f"delta={delta}")


def plot_kmean(current_cluster, delta):
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    fig, ax = plt.subplots()
    for index1 in range(k):
        pts = np.array([X[j] for j in range(len(X)) if current_cluster[j] == index1])
        ax.scatter(pts[:, 0], pts[:, 1], s=7, c=colors[index1])
    ax.scatter(C[:, 0], C[:, 1], marker="*", s=200, c='#C0FFEE')
    plt.title("delta=%.4f" % delta)
    plt.show()


while delta != 0:
    # for each point in X
    # choose the nearest center
    for i in range(len(X)):
        distances = dist(X[i], C)
        cluster = np.argmin(distances)
        clusters[i] = cluster
    C_old = deepcopy(C)
    # for each group of X
    # calculate new center
    for i in range(k):
        points = [X[j] for j in range(len(X)) if clusters[j] == i]
        C[i] = np.mean(points, axis=0)
    delta = dist(C, C_old, None)
    plot_kmean(clusters, delta)
