import tensorflow as tf

# 定義2維向量兩筆
x = tf.Variable(tf.random.uniform((2, 2)))
print(x)

with tf.GradientTape() as tape:
    y = 5 * x ** 2 + 6 * x + 4
    # 微分後10x+6
    diff_1 = tape.gradient(y, x)
print("x=", x.numpy(), sep="\n")
# 10*x+6
# 10*0.8370+6
print("diff1=", diff_1.numpy(), sep="\n")

W = tf.Variable(tf.random.uniform((1, 1)))
b = tf.Variable(tf.zeros((1,)))
#x = tf.random.uniform((1, 1))
x = tf.Variable(tf.random.uniform((1, 1)))
with tf.GradientTape() as tape:
    # y = xW+xb
    y = tf.matmul(x, W) + 2 * b
    # 對W b x偏微分 值放入grad1[0] grad1[1] grad1[2]
    grad1 = tape.gradient(y, [W, b, x])
print("W=", W.numpy())
print("b=", b.numpy())
print("x=", x.numpy())
print("y=", y.numpy())
# 對W偏微分 值為x
print(grad1[0].numpy())
# 對b偏微分 值為2
print(grad1[1].numpy())
# 對x偏微分 值為W
print(grad1[2].numpy())