# pykt0511 demo34
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from joblib import dump, load

# 從網路讀取原始資料
DATA_URL = "http://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data"
# 將原始csv轉換為python可以使用的矩陣　header=None代表原始資料沒有表頭　prefix='X'　將表頭定義為X1 X2 Xn
df = pd.read_csv(DATA_URL, header=None, prefix='X')
# 切割資料欄位前60筆為資料　61筆為推測屬性
data, labels = df.iloc[:, :-1], df.iloc[:, -1]
print(data.shape)
print(labels.shape)
# 置換表頭X60成為Label
print(df.columns)
df.rename(columns={'X60':'Label'}, inplace=True)
print(df.columns)

# 建立KNN物件　利用KNN分類　n_neighbors=3 除了自己再找兩個最近的鄰居
knn1 = KNeighborsClassifier(n_neighbors=3)
# 將資料分為訓練資料與測試驗證資料
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3)
# 執行KNN
knn1.fit(X_train, y_train)
y_predict = knn1.predict(X_test)
print("testing score=", knn1.score(X_test, y_test))
print(y_predict)
print(y_test)

result_cm1 = confusion_matrix(y_test, y_predict)
print(result_cm1)

scores = cross_val_score(knn1, data, labels, cv=5, groups=labels)
print(scores)

# make a directory data 如果覺得model不錯可以把它存起來
dump(knn1, "demo37.joblib")
knn2 = load("demo37.joblib")
y_predict2 = knn2.predict(X_test)
result2 = confusion_matrix(y_predict, y_predict2)
print(result2)
