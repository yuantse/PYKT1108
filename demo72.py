import tensorflow as tf
import keras
from keras import datasets
from matplotlib import pyplot as plt
from keras.layers import Dense, Flatten
import numpy as np

# 顯示衣著圖片
(train_images, train_labels,), (test_images, test_labels) = datasets.fashion_mnist.load_data()
class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
plt.imshow(train_images[1])
plt.colorbar()
plt.grid(False)
plt.show()

# 顯示25張衣著圖片
print(train_images[0])
train_images = train_images / 255.0
test_images = test_images / 255.0
# 利用OFFSET設定起始點
OFFSET = 0
# 定義畫布
plt.figure(figsize=(10, 8))
for i in range(25):
    # 畫子圖
    plt.subplot(5, 5, i + 1)
    # 把座標關掉
    plt.xticks([])
    plt.yticks([])
    # 顯示灰階
    plt.imshow(train_images[i + OFFSET], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i + OFFSET]])
plt.show()

# 利用Flatten將原始資料整成一行 取代之前的reshap及One Hot encoding
# 將nn layer模型用list[]包起來 取代之前的model.add
layers = [Flatten(input_shape=(28, 28)),
          Dense(128, activation='relu'),
          Dense(64, activation='relu'),
          Dense(10, activation='softmax')]
# 定義為sequential並把layer帶入
model = keras.Sequential(layers)
model.summary()
# 因為沒有真的執行reshape 必須使用sparse_categorical_crossentropy
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=10)
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

num_rows = 5
num_cols = 3
num_images = num_rows * num_cols

# 畫預測圖
def plot_image(i, predictions_array, true_label, image):
    # 訓練資料的原始分類test_label
    true_label = true_label[i]
    image = image[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    # 將圖片以灰階進行顯示
    plt.imshow(image, cmap=plt.cm.binary)
    # 將猜測機率最大的值取出
    predicted_label = np.argmax(predictions_array)
    # 如果猜對顯示藍色 猜錯顯示紅色
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'
    # 顯示猜測機率值%
    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100 * np.max(predictions_array),
                                         class_names[true_label]), color=color)


# 繪製猜測機率值
def plot_value_array(i, predictions_array, true_label):
    true_label = true_label[i]
    plt.grid(False)
    # 種類為0~9
    plt.xticks(range(10))
    plt.yticks([])
    # 只要有機率出現定義為灰色
    thisPlot = plt.bar(range(10), predictions_array, color='#888888')
    plt.ylim([0, 1])
    # 取出猜測機率值
    predicted_label = np.argmax(predictions_array)
    # 先定義最高機率值為紅色
    thisPlot[predicted_label].set_color('red')
    # 再定義最高機率值為藍色 如果猜對 會把紅色蓋掉
    thisPlot[true_label].set_color('blue')


# 將測試資料帶入模型進行預測
predictions = model.predict(test_images)
# 定義畫布
plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
# 畫n張圖 num_images = num_rows * num_cols
for i in range(num_images):
    # 定義子圖 原始圖(左 因為2*i+1)
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
    # 呼叫自訂義繪圖模組
    plot_image(i, predictions[i], test_labels, test_images)
    # 定義子圖 猜測機率圖(右 因為2*i+2)
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
    plot_value_array(i, predictions[i], test_labels)
# 避免蓋到其他的圖
plt.tight_layout()
plt.show()
