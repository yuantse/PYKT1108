from sklearn import datasets, svm
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

iris = datasets.load_iris()
# 將資料從4維降到2維
pca = PCA(n_components=2)
data = pca.fit_transform(iris.data)
print(data.shape, iris.data.shape)
datamax = data.max(axis=0)
datamin = data.min(axis=0)
print(datamax)
print(datamin)
n = 2000
# 建立meshgrid 當作輸入資料
X, Y = np.meshgrid(np.linspace(datamin[0], datamax[0], n),
                   np.linspace(datamin[1], datamax[1], n))

# 再用SVC進行分類
svc = svm.SVC()
svc.fit(data, iris.target)
# 利用先前建立的meshgrid進行預測
Z = svc.predict(np.c_[X.ravel(), Y.ravel()])
plt.contour(X, Y, Z.reshape(X.shape),
            levels=[-.5, .5, 1.5], colors=['r', 'g', 'b'])
for i, c in zip([0, 1, 2], ['r', 'g', 'b']):
    d = data[iris.target == i]
    plt.scatter(d[:, 0], d[:, 1], c=c)
plt.show()