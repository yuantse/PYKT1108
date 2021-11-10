import numpy as np
from sklearn.naive_bayes import GaussianNB

# 建立6筆資料
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
Y = np.array([1, 1, 1, 2, 2, 2])
# 建立NB物件
classifier = GaussianNB()
# 使用NB進行分析
classifier.fit(X, Y)
newX = [[1, 0], [0, 1], [-1, 0], [0, -1]]
print(classifier.predict(newX))

# 建立NB物件
classifier2 = GaussianNB()
# 使用NB進行分析, 並且告知Y的種類數量
classifier2.partial_fit(X, Y, np.unique(Y))
# 進行預測
print(classifier2.predict(newX))
# 加入新的資料進行訓練
classifier2.partial_fit([[0, 0]], [1])
# 再度預測
print(classifier2.predict(newX))
