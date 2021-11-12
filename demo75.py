import tensorflow as tf

# 定義原始圖片
imageSourceArray = tf.constant([1, 1, 1, 0, 0, 0] * 6, tf.float32)
print(imageSourceArray)
# 排列原始圖片為6*6
images = tf.reshape(imageSourceArray, [1, 6, 6, 1])
# 直的轉成橫的
images = tf.transpose(images, perm=[0, 2, 1, 3])
# 顯示原始圖片矩陣
print(images[0, :, :, 0])
# filterSourceArray = tf.constant([1, 0, -1] * 3, tf.float32)
# 定義filter(patch kernel)
filterSourceArray = tf.constant([1, 0, -1] * 3, tf.float32)
# 排列filter為3*3
filter = tf.reshape(filterSourceArray, [3, 3, 1, 1])
# 直的轉成橫的
filter = tf.transpose(filter, perm=[1, 0, 2, 3])
# 顯示filter矩陣
print(filter[:, :, 0, 0])
# 進行convolution
conv = tf.nn.conv2d(images, filter, [1, 1, 1, 1], padding='VALID')
# 顯示結果
convResult = conv.numpy()
print(convResult.shape)
print(convResult[0, :, :, 0])