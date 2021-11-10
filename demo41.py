import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA


# 讀資料
iris = datasets.load_iris()
X = iris.data
species = iris.target

# 將資料維度從4皆降為3皆
X_reduced = PCA(n_components=3).fit_transform(iris.data)
print(X.shape, X_reduced.shape)


# 定義繪圖框1張 9*9
fig = plt.figure(1, figsize=(9, 9))

# 帶入3D座標至繪圖框 並定義視角
ax = Axes3D(fig, elev=-150, azim=110)
# 分別將三種類別以不同顏色顯現出來
ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=species)
# 給予座標標示
ax.set_xlabel("first eigen")
ax.set_ylabel("second eigen")
ax.set_zlabel("third eigen")
plt.show()
