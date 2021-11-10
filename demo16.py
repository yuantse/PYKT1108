import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
# polynomial
from sklearn.preprocessing import PolynomialFeatures

x = np.array([5, 15, 25, 35, 45, 55]).reshape((-1, 1))
y = np.array([15, 11, 2, 8, 25, 32])
plt.plot(x, y)
plt.scatter(x, y)
plt.show()
regression1 = LinearRegression()
regression1.fit(x, y)

x_sequence = np.array(np.arange(5, 55, 0.1)).reshape(-1, 1)
plt.plot(x, y)
plt.scatter(x, y)
plt.plot(x, regression1.coef_ * x + regression1.intercept_)
print("linear regression score=", regression1.score(x, y))
plt.title("linear regression score=%.2f" % regression1.score(x, y))
plt.show()