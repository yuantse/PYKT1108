import matplotlib.pyplot as plt
import numpy as np

range1 = [-1, 3]
p = np.array([3])

print(p*range1+5)
# [3*(-1)+5, 3*3+5] --> [-2, 14]

# plt.plot(scale x, scale y, c='green')
plt.plot(range1, p*range1+5, c='green')
plt.show()