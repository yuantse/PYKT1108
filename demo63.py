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
# total layer = (default k-fold)5*(3 layer)3*(optimizers)2*(inits)2*(epochs)3*(batches)3
grid = GridSearchCV(estimator=model, param_grid=param_grid)
grid_result = grid.fit(inputList, resultList)

# 印出最好的參數組合 最好的預測值
grid_result.best_params_, grid_result.best_score_


# 印出所有的運算結果
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("param:{} ==> score=>{}, std=>{}".format(mean,stdev,param))
