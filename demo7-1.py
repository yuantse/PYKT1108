import matplotlib.pyplot as plt
from sklearn import linear_model

# y = coefficient*x + intercept
regression1 = linear_model.LinearRegression()
features = [[1], [2], [3]]
values = [1, 5, 9]
plt.scatter(features, values, c='green')
plt.show()
regression1.fit(features, values)
print("coefficient=", regression1.coef_)
print("intercept=", regression1.intercept_)
range1 = [min(features), max(features)]
print(regression1.coef_ * range1 + regression1.intercept_)
print("score=", regression1.score(features, values))
plt.plot(range1, regression1.coef_ * range1 + regression1.intercept_, c='gray')
plt.scatter(features, values, c='green')
plt.show()
