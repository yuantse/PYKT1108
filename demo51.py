import tensorflow as tf
from datetime import datetime

# 計算三角形公式
# 手動建一個目錄logs 比較有無@tf.function的差異

@tf.function
def computeArea(sides):
    a = sides[:, 0]
    b = sides[:, 1]
    c = sides[:, 2]
    s = (a + b + c) / 2
    areaSquare = s * (s - a) * (s - b) * (s - c)
    return areaSquare ** 0.5

stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = 'logs/demo51/%s' % stamp
writer = tf.summary.create_file_writer(logdir)
tf.summary.trace_on(graph=True, profiler=True)

print(computeArea(tf.constant([[3.0, 4.0, 5.0], [6.0, 6.0, 6.0], [6.0, 8.0, 10.0]])))
print(computeArea(tf.constant([[3.0, 4.0, 5.0], [6.0, 6.0, 6.0], [6.0, 8.0, 10.0]])).numpy())

with writer.as_default():
    tf.summary.trace_export(name='trace_graph', step=0, profiler_outdir=logdir)