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
# 二元劃分問題建議使用binary_crossentropy 梯度下降法最佳參數為adam 測量指標為accuracy
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# param計算
# 第一層 (8+1)*14 = 129 +1代表常數b
# 第二層 (14+1)*8 = 120
# 第三層 (8+1)*1 = 9
model.summary()

# 開始訓練
model.fit(inputList, resultList, epochs=200, batch_size=20)
# 評估結果
scores = model.evaluate(inputList, resultList)
print("type(scores)")
print(type(scores))
print("scores")
print(scores)
print("model.metrics_names")
print(model.metrics_names)
for s, m in zip(scores, model.metrics_names):
    print("{}={}".format(m, s))