import matplotlib.pyplot as plt
import numpy as np

# y= ax+b
b = 5
# linespace含頭尾
a = np.linspace(5, -5, 11)
# arrange支援浮點數 含頭不含尾
x = np.arange(-10, 10, 0.1)
print(x)
for a1 in a:
    y = a1 * x + b
    plt.plot(x, y, label=f"y={a1}x+{b}")
    # 劃出圖示說明
    plt.legend(loc=2)
# 畫作標線
plt.axhline(0, color='black')
plt.axvline(0, color='black')
plt.title("demo4 figure")
plt.xlabel("label for x")
plt.ylabel("label for y")
# plt.xticks([])
# plt.yticks([])
plt.show()
