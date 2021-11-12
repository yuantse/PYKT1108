import numpy as np
from tensorflow import nn

scores = [3.0, 1.0, 2.0]

# 把值轉換成0~1
def normalRatio(x):
    x = np.array(x)
    return x / np.sum(x)


def mySoftMax(x):
    x = np.array(x)
    return np.exp(x) / np.sum(np.exp(x))


print(normalRatio(scores))
# 手動計算softmax
print(mySoftMax(scores))
# 用tensorflow計算softmax
print(nn.softmax(scores).numpy())