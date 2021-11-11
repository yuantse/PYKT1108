import numpy as np
from keras import layers, models
from keras.datasets import imdb

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