import tensorflow as tf

time = tf.Variable(5.)

# 計算速度加速度 二次微分
with tf.GradientTape() as outer_tape:
    with tf.GradientTape() as inner_tape:
        position = 4.9 * time ** 2
        speed = inner_tape.gradient(position, time)
        print("speed=", speed)
    acc = outer_tape.gradient(speed, time)
    print("accelerator=", acc)