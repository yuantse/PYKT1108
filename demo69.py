from matplotlib import pyplot as plt
import tensorflow as tf
import keras
import numpy

# 數字影像資料
(train_images, train_labels), (test_images, test_labels) = keras.datasets.mnist.load_data()
print("image shape", train_images.shape, test_images.shape)
print("label shape", train_labels.shape, test_labels.shape)
# 計算各個數字出現的次數
print(numpy.unique(train_labels, return_counts=True))
print(numpy.unique(test_labels, return_counts=True))


# 劃出原始訓練資料圖像
def plotImage(index):
    plt.title("the image is %d" % train_labels[index])
    plt.imshow(train_images[index], cmap='binary')
    plt.show()


plotImage(10005)


# 劃出原始驗證資料圖像
def plotTestImage(index):
    plt.title("the test image is %d" % test_labels[index])
    plt.imshow(test_images[index])
    plt.show()


plotTestImage(3000)
