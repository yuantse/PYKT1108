from numpy import array
from sklearn.decomposition import PCA

# 建立4個向量
A = array([[1, 2, 3], [3, 4, 5], [5, 6, 7], [7, 8, 9]])
print(A)

# 定義PCA為2皆
pca1 = PCA(2)
# 將A從4個向量降為2個向量
pca1.fit(A)
print(pca1)
print(pca1.components_)
# 顯示向量重要性
print(pca1.explained_variance_)
print(pca1.explained_variance_ratio_)

# 直接將向量從3維轉2維
B = pca1.transform(A)
print(B)