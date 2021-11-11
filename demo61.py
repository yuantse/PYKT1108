import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import StratifiedKFold

FILENAME = 'diabetes.csv'
dataset1 = np.loadtxt(FILENAME, delimiter=',', skiprows=1)
print(type(dataset1))
print(dataset1.shape)
inputList = dataset1[:, :8]
resultList = dataset1[:, 8]
print(inputList.shape)
print(resultList.shape)

# 定義k-fold 將訓練資料切割成K等分 依序訓練除了其中一分之外的所有資料 剩下的一份當作評估資料
# 若切割成5份 則會進行5輪運算
fiveFold = StratifiedKFold(n_splits=5, shuffle=True)
totalScores = []


def createModel():
    model = Sequential()
    model.add(Dense(14, input_dim=8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # model.summary()
    return model


for train, test in fiveFold.split(inputList, resultList):
    print("init a run")
    m = createModel()
    # verbose=0 不顯示每次運算的結果
    # 開始訓練
    m.fit(inputList[train], resultList[train], epochs=200, batch_size=20, verbose=0)
    # 評估結果
    scores = m.evaluate(inputList[test], resultList[test])
    totalScores.append(scores[1] * 100)
print("total score mean={}, std={}".format(np.mean(totalScores), np.std(totalScores)))
