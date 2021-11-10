import tensorflow as tf

vectors = [3.0, -1.0, 2.4, 5.9, 0.0001, -0.0005, 8.5, 100, 30000, 0.49, 0.51, 0.001]
# relu function
print(tf.nn.relu(vectors).numpy())
# sigmoid function
print(tf.nn.sigmoid(vectors).numpy())

