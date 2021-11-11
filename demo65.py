import numpy
from keras.datasets import imdb
from matplotlib import pyplot as plt

# 讀影評資料庫imdb
(X_train, y_train), (X_test, y_test) = imdb.load_data()
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
# 將資料接起來 axis=0 代表直的接
X = numpy.concatenate((X_train, X_test), axis=0)
y = numpy.concatenate((y_train, y_test), axis=0)
print(X.shape)
print(y.shape)
# 從結果(y)中計算0出現的次數 1出現的次數
print(numpy.unique(y, return_counts=True))

# numpy.hstack 將原始資料(字串)接起來 numpy.unique 篩選掉重複的字 len 計算總字彙量
print(len(numpy.unique(numpy.hstack(X))))

# 算出影評平均字數與標準差
# 標準差大 代表有些影評很長
result = [len(x) for x in X]
print("comments mean={}, std={}".format(numpy.mean(result), numpy.std(result)))


plt.subplot(121)
# 顯示盒圖
plt.boxplot(result)
plt.subplot(122)
# 顯示長條圖
plt.hist(result)
plt.show()