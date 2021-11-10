import tensorflow as tf

# @tf.function 可最佳化function的運算過程
@tf.function
def add(p, q):
    return tf.add(p, q)


print(add([1, 2, 3], [4, 5, 6]).numpy())