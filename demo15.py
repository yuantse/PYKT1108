# same as pykt0511 demo14
import numpy as np
from sklearn import linear_model, datasets

diabetes = datasets.load_diabetes()
print(type(diabetes), diabetes.data.shape, diabetes.target.shape)
dataForTest = -60

data_train = diabetes.data[:dataForTest]
target_train = diabetes.target[:dataForTest]
print("data trained:", data_train.shape)
print("target trained:", target_train.shape)
data_test = diabetes.data[dataForTest:]
target_test = diabetes.target[dataForTest:]
print("data test:", data_test.shape)
print("target test:", target_test.shape)
# start making regression
regression1 = linear_model.LinearRegression()
regression1.fit(data_train, target_train)
print(regression1.coef_)
print(regression1.intercept_)

print("score=", regression1.score(data_test, target_test))

for i in range(dataForTest, 0):
    data1 = np.array(data_test[i]).reshape(1, -1)
    print("predict={:.2f}, actual={:.2f}".format(regression1.predict(data1)[0], target_test[i]))

mean_square_error = np.mean((regression1.predict(data_test) - target_test) ** 2)
print(mean_square_error)