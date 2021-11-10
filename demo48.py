# tensorflow v1
import tensorflow as tf
import numpy as np

tf.compat.v1.disable_eager_execution()
a = np.array([5, 3, 8])
b = np.array([3, -1, 2])
print(a + b)
print(np.add(a, b))

t1 = tf.constant([5, 3, 8])
t2 = tf.constant([3, -1, 2])
print(t1 + t2)
# print((t1 + t2).numpy())
print(tf.add(t1, t2))
# print(np.add(t1, t2))
with tf.compat.v1.Session() as session:
    print(session.run(tf.add(t1, t2)))
    print(session.run(t1 + t2))