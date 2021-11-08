# 手動regression

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets

regressionData = datasets.make_regression(100, 1, noise=5)
plt.scatter(regressionData[0], regressionData[1], c='red', marker='^')
init_m = 10
init_b = 10
learning_rate = 0.1
range1 = [-5, 5]
plt.plot(range1, init_m * np.array(range1) + init_b)
plt.show()


def cost(m, b, X, Y):
    N = len(X)
    cost = 0
    for i in range(N):
        cost += (Y[i] - (m * X[i] + b)) ** 2
    return cost / N


init_cost = cost(init_m, init_b, regressionData[0], regressionData[1])
print("init cost=%.2f" % init_cost[0])


def update_weights(m, b, X, Y, learning_rate):
    m_deriv = 0
    b_deriv = 0
    N = len(X)
    for i in range(N):
        m_deriv += -2 * X[i] * (Y[i] - (m * X[i] + b))
        b_deriv += -2 * (Y[i] - (m * X[i] + b))
    m -= (m_deriv / float(N)) * learning_rate
    b -= (b_deriv / float(N)) * learning_rate
    return m, b