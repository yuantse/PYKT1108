import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import KFold, cross_val_score
from keras.wrappers.scikit_learn import KerasClassifier


# 讀原始資料
FILE_NAME = 'iris.data'
df1 = pd.read_csv(FILE_NAME, header=None)
dataset = df1.values
features = dataset[:, 0:4].astype(float)
labels = dataset[:, 4]
# 資料欄位
print(features.shape)
# 種類欄位
print(labels.shape)
print(np.unique(labels, return_counts=True))
print(df1.describe())

# 定義編碼器
encoder = LabelEncoder()
# 將原始資料編譯成0 1 2
encoder.fit(labels)
encoded_Y = encoder.transform(labels)
# 計算0 1 2 出現的次數
print(np.unique(encoded_Y, return_counts=True))
# 將種類0 1 2轉換成one hot encoding 100, 010, 001
dummy_y = np_utils.to_categorical(encoded_Y)
# 印出encode後的shape
print(dummy_y.shape)
# 印出encode後的前5筆種類
print(dummy_y[:5])


# 定義訓練模型
def baselineModel():
    model = Sequential()
    model.add(Dense(8, input_dim=4, activation='relu'))
    # 多種類輸出建議使用'softmax'
    model.add(Dense(3, activation='softmax'))
    model.summary()
    model.compile(loss="categorical_crossentropy", optimizer='adam',
                  metrics=['accuracy'])
    return model

# 將model用KerasClassifier包起來
estimator = KerasClassifier(build_fn=baselineModel, epochs=200, batch_size=10, verbose=1)
# 定義k-fold分為3群
kfold = KFold(n_splits=3, shuffle=True)
# 執行訓練 estimator為訓練模型 features為資料參數 dummy_y為種類 cross validation用kfold
result = cross_val_score(estimator, features, dummy_y, cv=kfold)
print("acc={}, std={}".format(result.mean(), result.std()))
