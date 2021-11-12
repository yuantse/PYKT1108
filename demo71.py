import tensorflow as tf
import keras
import numpy as np
from keras import layers, models
from matplotlib import pyplot as plt
import pandas as pd

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

FLATTEN_DIM = 28 * 28
TRAINING_SIZE = len(train_images)
TEST_SIZE = len(test_images)
# 把資料攤平成1個維度 28*28 -> 784
trainImages = np.reshape(train_images, (TRAINING_SIZE, FLATTEN_DIM))
testImages = np.reshape(test_images, (TEST_SIZE, FLATTEN_DIM))
print(type(trainImages[0]), trainImages.shape, trainImages[0].shape)

# 將圖數值標準化0~1 使多項式參數a接近0~1 (ax+b)
trainImages = trainImages.astype(np.float32)
testImages = testImages.astype(np.float32)
trainImages /= 255
testImages /= 255
print(trainImages[1])

# 將Labels轉換成0~9 one hot encoding
NUM_DIGITS = 10
trainLabels = keras.utils.np_utils.to_categorical(train_labels, NUM_DIGITS)
testLabels = keras.utils.np_utils.to_categorical(test_labels, NUM_DIGITS)
print(trainLabels[:5])

model = models.Sequential()
model.add(layers.Dense(128, activation=tf.nn.relu, input_shape=(FLATTEN_DIM,)))
model.add(layers.Dense(10, activation=tf.nn.softmax))
# 分類問題建議使用categorical_crossentropy
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
# 將運算過程顯示在tensorboard上
board = keras.callbacks.TensorBoard(log_dir='logs/demo71/', histogram_freq=0,
                                    write_graph=True, write_images=True)
# model.fit(trainImages, trainLabels, epochs=20)
model.fit(trainImages, trainLabels, epochs=20, callbacks=[board])

# run in jupyter-notebook
# 預測測試資料
predictResult = model.predict(testImages)
# 顯示前5筆
predictResult[:5]
# 將結果轉換成數字0~9
predict = np.argmax(predictResult, axis=-1)
predict[:5]

# 顯示loss accuracy
loss, accuracy = model.evaluate(testImages, testLabels)
loss, accuracy

# 顯示測試圖片
def plotTestImage(index):
    plt.title("the test image is %d"%test_labels[index])
    plt.imshow(test_images[index])
    plt.show()
plotTestImage(5)

# 再訓練一次 validation_split=0.1 10%資料當作驗證validation資料
trainHistory = model.fit(trainImages, trainLabels, epochs=20, validation_split=0.1)

# 顯示training 與 validation曲線
plt.plot(trainHistory.history['accuracy'],color='red')
plt.plot(trainHistory.history['val_accuracy'], color='green')
plt.legend(['training', 'validation'])

# 顯示 cross矩陣 label=predict 代表預測正確
pd.crosstab(test_labels, predict, rownames=['label'], colnames=['predict'])
# 前20個預測結果
measure1 = pd.DataFrame({'label':test_labels, 'predict':predict})
measure1[:20]

# 顯示標示為7 錯誤為2
measure1[(measure1.label==7) & (measure1.predict==2)]

# 將顯示標示為7 錯誤為2的存起來
error_2_7 = measure1[(measure1.label==7) & (measure1.predict==2)]

# 顯示顯示標示為7 錯誤為2的圖片
print(error_2_7.index)
for i in error_2_7.index:
    plotTestImage(i)
# run in jupyter-notebook
