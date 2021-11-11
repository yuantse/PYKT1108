import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split

FILENAME = 'diabetes.csv'
dataset1 = np.loadtxt(FILENAME, delimiter=',', skiprows=1)
print(type(dataset1))
print(dataset1.shape)
inputList = dataset1[:, :8]
resultList = dataset1[:, 8]
print(inputList.shape)
print(resultList.shape)

print(np.unique(resultList, return_counts=True))
# 隨機切割資料 但結果資料 訓練資料 測試資料 每次0 1 比率相同 訓練
feature_train, feature_test, label_train, label_test = train_test_split(inputList, resultList,
                                                                        test_size=0.2, stratify=resultList)

for d in [resultList, label_train, label_test]:
    classes, counts = np.unique(d, return_counts=True)
    for cl, co in zip(classes, counts):
        print("{}==>{:.2f}".format(int(cl), co / sum(counts)))