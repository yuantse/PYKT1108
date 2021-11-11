import numpy as np
from keras import layers, models
from keras.datasets import imdb
import matplotlib.pyplot as plt

# 詞彙量只取15000
MAX_WORDS = 15000
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=MAX_WORDS)
print(train_data[0])
print(max([max(sequence) for sequence in train_data]))

# 讀入字典, 每個數字代表一個單字
word_index = imdb.get_word_index()
print(type(word_index))
# 顯示前10筆字典內容
print(list(word_index.items())[:10])
# 將數字轉為文字
reverse_word_index = dict([(v, k) for (k, v) in word_index.items()])
# 顯示前5筆影評
for i in range(5):
    # 前3個字為保留字
    decoded_review = ' '.join([reverse_word_index.get(i - 3, "?") for i in train_data[i]])
    print(decoded_review)

# 將原始資料轉換成25000*15000的矩陣
'''
        15000(字彙) 有出現就填一
     |----------------
     |1 0 0 1 0 0 0 1
影評  |0 0 0 1 0 1 0 0
25000|
     |
     |
'''
def vectorize_sequence(sequences, dimension=MAX_WORDS):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results


# 將影評轉為矩陣
X_train = vectorize_sequence(train_data)
X_test = vectorize_sequence(test_data)
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')
# 印出第一筆資料
print(X_train[0])
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# 建立訓練模型
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(MAX_WORDS,)))
model.add(layers.Dense(1, activation='sigmoid'))
model.summary()
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

# 開始訓練
history = model.fit(X_train, y_train, epochs=30, batch_size=256,
                    validation_data=(X_test, y_test))

# 顯示準確度
history_dict = history.history
accuracy = history_dict['accuracy']
val_accuracy = history_dict['val_accuracy']
epochs = range(1, len(accuracy)+1)
plt.plot(epochs, accuracy, 'g--', label='accuracy')
plt.plot(epochs, val_accuracy, 'b-', label='validation accuracy')
plt.legend()

# 顯示loss function
loss = history_dict['loss']
val_loss = history_dict['val_loss']
plt.plot(epochs, loss, 'g--', label='loss')
plt.plot(epochs, val_loss, 'b-', label='validation loss')
plt.legend()
