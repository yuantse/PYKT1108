import tensorflow as tf


# x = tf.Variable(0.)
x = tf.Variable(10.)
with tf.GradientTape() as tape:
    y = 2 * x + 3
    # 對x微分
    diff_x = tape.gradient(y, x)
    print(diff_x.numpy())

with tf.GradientTape() as tape:
    y2 = 2 * x ** 2 + 3 * x + 4
    # 對x微分
    diff_x_2 = tape.gradient(y2, x)
    print(diff_x_2.numpy())