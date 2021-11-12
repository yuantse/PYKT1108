import tensorflow as tf
import pandas as pd
import keras
from sklearn.preprocessing import LabelBinarizer
from keras import callbacks
from keras import layers, models

# 讀資料
csv1 = pd.read_csv("data/bmi.csv")
# 將資料標準化到接近0~1
csv1['height'] = csv1['height'] / 200
csv1['weight'] = csv1['weight'] / 100
# 顯示前10筆資料
print(csv1[:10])
# 利用LabelBinarizer() 將label進行One Hot encoding
encoder = LabelBinarizer()
transformedLabel = encoder.fit_transform(csv1['label'])
# 顯示前10筆label
print(csv1['label'][:10])
# 顯示One Hot encoding後的label
print(transformedLabel[:10])

# 取25000筆為測試資料
test_csv = csv1[25000:]
test_pat = test_csv[['weight', 'height']]
test_ans = transformedLabel[25000:]

# 取25000筆為訓練資料
train_csv = csv1[:25000]
train_pat = train_csv[['weight', 'height']]
train_ans = transformedLabel[:25000]

# 定義訓練模型
model = models.Sequential()
model.add(layers.Dense(10, activation='relu', input_shape=(2,)))
model.add(layers.Dense(3, activation='softmax'))
model.summary()
# 因為已進行one hot encoding 選擇categorical_crossentropy
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
board = callbacks.TensorBoard(log_dir='logs/demo74')
# 開始訓練
model.fit(train_pat, train_ans, batch_size=50, epochs=100, verbose=1,
          validation_data=(test_pat, test_ans), callbacks=[board])
# 評估成果
score = model.evaluate(test_pat, test_ans, verbose=0)
print(score)