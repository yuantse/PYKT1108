import matplotlib.pyplot as plt
import numpy as np

# y= ax+b
b = np.linspace(5, -5, 11)
# 從5~-5產生11個值, 含頭尾
print(b)
a = 3
x = np.arange(-10, 10, 0.1)
# 從-10~(10-0.1) 每0.1產生一個值, 含頭部含尾
print(x)

for b1 in b:
    y = a * x + b1
    plt.plot(x, y, label=f"y={a}x+{b1}")
    plt.legend(loc=2)
plt.axhline(0, color='black')
plt.axvline(0, color='black')
plt.title("demo2 figure")
plt.xlabel("label for x")
plt.ylabel("label for y")
# plt.xticks([])
# plt.yticks([])
plt.show()