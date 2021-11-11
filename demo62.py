from keras.models import save_model, load_model
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

FILENAME = 'diabetes.csv'
dataset1 = np.loadtxt(FILENAME, delimiter=',', skiprows=1)
print(type(dataset1))
print(dataset1.shape)
inputList = dataset1[:, :8]
resultList = dataset1[:, 8]
print(inputList.shape)
print(resultList.shape)


def createModel():
    m = Sequential()
    m.add(Dense(14, input_dim=8, activation='relu'))
    m.add(Dense(8, activation='relu'))
    m.add(Dense(1, activation='sigmoid'))
    m.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    m.summary()
    return m


# 將model包裝成Classifier
model = KerasClassifier(build_fn=createModel, epochs=200, batch_size=20, verbose=0)
fiveFold = StratifiedKFold(n_splits=5, shuffle=True)
# cv=fiveFold 指定cross validation方式, 每次執行會重新建立一次model
results = cross_val_score(model, inputList, resultList, cv=fiveFold)
print("mean=%.3f, std=%.3f" % (results.mean(), results.std()))