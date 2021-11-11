from keras.models import save_model, load_model
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV


FILENAME = 'diabetes.csv'
dataset1 = np.loadtxt(FILENAME, delimiter=',', skiprows=1)
print(type(dataset1))
print(dataset1.shape)
inputList = dataset1[:, :8]
resultList = dataset1[:, 8]
print(inputList.shape)
print(resultList.shape)


def createModel(optimizer='adam', init='uniform'):
    m = Sequential()
    m.add(Dense(14, input_dim=8, kernel_initializer=init, activation='relu'))
    m.add(Dense(8, activation='relu'))
    m.add(Dense(1, activation='sigmoid'))
    m.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    m.summary()
    return m


# 將model包裝成Classifier
model = KerasClassifier(build_fn=createModel, verbose=0)
# 定義各種model參數
optimizers = ['rmsprop', 'adam']
inits = ['normal', 'uniform']
epochs = [50, 100, 150]
batches = [5, 10, 15]
param_grid = dict(optimizer=optimizers, epochs=epochs, batch_size=batches,
                  init=inits)
# 跑各種model參數並cross validation
grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid_result = grid.fit(inputList, resultList)
