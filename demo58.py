import numpy as np
from keras.models import Sequential
from keras.layers import Dense

FILENAME = 'diabetes.csv'
dataset1 = np.loadtxt(FILENAME, delimiter=',', skiprows=1)
print(type(dataset1))
print(dataset1.shape)
inputList = dataset1[:, :8]
resultList = dataset1[:, 8]
print(inputList.shape)
print(resultList.shape)

# sequential 每層的輸入等於下一層的輸出
model = Sequential()
# input_dim 視資料維度而定
# 第一層 輸入8個維度 輸出14個維度
model.add(Dense(14, input_dim=8, activation='relu'))
# 輸出8個維度
model.add(Dense(8, activation='relu'))
# 輸出1個維度(結果)
model.add(Dense(1, activation='sigmoid'))
# model.compile
model.summary()