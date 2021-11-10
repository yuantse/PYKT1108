# pykt0511 demo32
import numpy as np
from sklearn.neighbors import NearestNeighbors

# 產生6組點
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])

# 用KNN分析 n_neighbors=2 參考自己跟一個鄰居
neighbors = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(X)
# 回傳KNN分析後的結果參數
distances, indices = neighbors.kneighbors(X, return_distance=True)
# 印出每個點與最近鄰居的距離
print(distances)
# 印出每個點與其最近距離的點
print(indices)
# 印出每個點與其最近距離參照矩陣圖(類似運動賽事對戰表)
print(neighbors.kneighbors_graph(X).toarray())
