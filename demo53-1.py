import tensorflow as tf

# 定義常數
input_const = tf.constant(3.)
with tf.GradientTape() as tape:
    # 觀察常數再向量中的變化 使其可微分
    tape.watch(input_const)
    result = tf.square(input_const)
    g1 = tape.gradient(result, input_const)
    print(g1)

W = tf.Variable(tf.random.uniform((1, 1)))
b = tf.Variable(tf.zeros((1,)))
x = tf.random.uniform((1, 1))
# x = tf.Variable(tf.random.uniform((1, 1)))
with tf.GradientTape() as tape:
    # 觀察常數再向量中的變化 使其可微分
    tape.watch(x)
    y = tf.matmul(x, W) + 2 * b
    # 對x微分 值為W
    grad1 = tape.gradient(y, x)
print("x=", x.numpy())
print("w=", W.numpy())
print("b=", b.numpy())
print("y=", y.numpy())
print(grad1)