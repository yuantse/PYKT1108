import tensorflow as tf

# disable tf2 feature, tf1 behavior is back
tf.compat.v1.disable_eager_execution()
t1 = tf.constant("hello tensorflow")
print(t1)
session1 = tf.compat.v1.Session()
print(session1.run(t1))
session1.close()

with tf.compat.v1.Session() as session2:
    print("using with as:", session2.run(t1))