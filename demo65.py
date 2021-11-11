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