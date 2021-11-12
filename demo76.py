import tensorflow as tf
from keras import layers, models
from keras.datasets import mnist
from keras.utils.np_utils import to_categorical

# CNN -> convolution + NN

# 建立sequential
model = models.Sequential()
# same 代表經過filter後像素不會變少 因此原始檔案旁邊要補值
# model.add(layers.Conv2D(32, (3, 3), activation=tf.nn.relu, input_shape=(28, 28, 1), padding='same'))
# 32代表產生32個feature map
# (3, 3)帶表filter為3*3 padding預設為valid經過filter(kernel patch)像素會變少
# input_shape 像素28*28 1代表黑白(長寬深) input_shape視原始資料而定
model.add(layers.Conv2D(32, (3, 3), activation=tf.nn.relu, input_shape=(28, 28, 1), padding='valid'))
#　從周邊2*2取最大的
model.add(layers.MaxPooling2D((2, 2)))
# extract from feature map
# 從feature map再做一次
model.add(layers.Conv2D(32, (3, 3), activation=tf.nn.relu, padding='valid'))
model.add(layers.MaxPooling2D((2, 2)))
# 從feature map再做一次　但不maxpolling
model.add(layers.Conv2D(32, (3, 3), activation=tf.nn.relu, padding='valid'))
# 把資料攤平 再用基本nn運算
model.add(layers.Flatten())
# 降維度
model.add(layers.Dense(32, activation=tf.nn.relu))
# 再降維度到10 因為原始資料有10種(0~9)
model.add(layers.Dense(10, activation=tf.nn.softmax))
model.summary()

# 讀訓練資料
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
# reshape
train_images = train_images.reshape((60000, 28, 28, 1))
# 像素表準化維0~1
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255
# 將結果分類
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# 開始訓練運算
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5, batch_size=32)
# 評估預測模型
score = model.evaluate(test_images, test_labels, verbose=0)
print(score)