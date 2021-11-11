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
# 隨機切割資料 但訓練資料 測試資料 類別0,1的比率相同
feature_train, feature_test, label_train, label_test = train_test_split(inputList, resultList,
                                                                        test_size=0.2, stratify=resultList)

for d in [resultList, label_train, label_test]:
    classes, counts = np.unique(d, return_counts=True)
    for cl, co in zip(classes, counts):
        print("{}==>{:.2f}".format(int(cl), co / sum(counts)))

# 建立模型
model = Sequential()
model.add(Dense(14, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# validation_data=(feature_test, label_test) 指定驗證資料集
model.fit(feature_train, label_train, validation_data=(feature_test, label_test), epochs=200, batch_size=20)
scores = model.evaluate(feature_test, label_test)

for s, m in zip(scores, model.metrics_names):
    print("{}={}".format(m, s))