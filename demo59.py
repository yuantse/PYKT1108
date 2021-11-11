from keras.models import save_model, load_model
import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# 手動建一個目錄models

FILENAME = 'diabetes.csv'
dataset1 = np.loadtxt(FILENAME, delimiter=',', skiprows=1)
print(type(dataset1))
print(dataset1.shape)
inputList = dataset1[:, :8]
resultList = dataset1[:, 8]
print(inputList.shape)
print(resultList.shape)


def createModel():
    m = Sequential()
    m.add(Dense(14, input_dim=8, activation='relu'))
    m.add(Dense(8, activation='relu'))
    m.add(Dense(1, activation='sigmoid'))
    m.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    m.summary()
    return m

# 呼叫createModel()並建立model
model = createModel()
# 開始訓練
model.fit(inputList, resultList, epochs=200, batch_size=20)
MODEL_PATH = 'models/demo59'
# 儲存訓練後的model
save_model(model, MODEL_PATH)
scores = model.evaluate(inputList, resultList)
print("trained model:", scores)

# 建立model但不訓練
model2 = createModel()
scores2 = model2.evaluate(inputList, resultList)
print("not trained model:", scores2)

# 讀取先前訓練過的model並使用
model3 = load_model(MODEL_PATH)
scores3 = model3.evaluate(inputList, resultList)
print("loaded model:", scores3)
