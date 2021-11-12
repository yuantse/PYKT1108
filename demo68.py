# 說明累加對大數的影響
# 1000 OK
# 10000 upper
y = 1000


def calculate(x):
    for i in range(0, 1000000):
        x += 0.0000001
    x -= 0.1
    return x


print('result=%.6f' % calculate(y))