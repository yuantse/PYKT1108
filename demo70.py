from keras.utils import np_utils

# one hot encoding 範例
orig = [4, 6, 8]
# 資料維度(結果)為15
NUM_DIGITS = 15

for o in orig:
    print("orig={}, shift={}".format(o, np_utils.to_categorical(o, NUM_DIGITS)))
