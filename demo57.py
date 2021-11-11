import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# 產生兩組點
num_samples_per_class = 1000
negative_samples = np.random.multivariate_normal(mean=[0, 4], cov=[[1, 0.5], [0.5, 1]], size=num_samples_per_class)
positive_samples = np.random.multivariate_normal(mean=[-4, 0], cov=[[1, 0.5], [0.5, 1]], size=num_samples_per_class)

inputs = np.vstack((negative_samples, positive_samples)).astype(np.float32)
targets = np.vstack(
    (np.zeros((num_samples_per_class, 1), dtype="float32"),
     np.ones((num_samples_per_class, 1), dtype="float32")))
plt.scatter(inputs[:, 0], inputs[:, 1], c=targets[:, 0])
plt.show()

# 輸入為兩個維度 (x, y)
input_dim = 2
# 輸出為種類
output_dim = 1

# y=Wx+b
# 定義W, b變數
W = tf.Variable(initial_value=tf.random.uniform(shape=(input_dim, output_dim)))
b = tf.Variable(initial_value=tf.zeros(shape=(output_dim,)))


def model(inputs):
    # 代表y=Wx+b, x = inputs
    return tf.matmul(inputs, W) + b


def square_loss(targets, predictions):
    per_sample_loss = tf.square(targets - predictions)
    return tf.reduce_mean(per_sample_loss)

# 梯度下降法
# [theta j] = [theta i] - [learning rate]*[對theta i微分的總和 這個範例是對W b微分]
def trainig_step(inputs, targets):
    learning_rate = 0.05
    with tf.GradientTape() as tape:
        predictions = model(inputs)
        loss = square_loss(predictions, targets)
    gradient1_w, gradient1_b = tape.gradient(loss, [W, b])
    W.assign_sub(gradient1_w * learning_rate)
    b.assign_sub(gradient1_b * learning_rate)
    return loss


for step in range(40):
    loss = trainig_step(inputs, targets)
    print("loss for step{}:{:.4f}".format(step, loss))

# 預測並劃出分類線
predictions = model(inputs)
x = np.linspace(-2.5, -2, 100)
y = -W[0] / W[1] * x + (0.5 - b) / W[1]
plt.plot(x, y, "-r")
# plt.scatter(inputs[:, 0], inputs[:, 1], c=predictions[:, 0] > 0.5)
plt.scatter(inputs[:, 0], inputs[:, 1], c=targets[:, 0])
plt.show()