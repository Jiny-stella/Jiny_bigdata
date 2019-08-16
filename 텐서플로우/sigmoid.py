# sigmoid.py

import math
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1./(1. + math.e**-z)

# print(math.e)  # e = 2.718281828459045

print(sigmoid(-100))  # 0에 수렴
print(sigmoid(-10))
print(sigmoid(0))  # 0.5
print(sigmoid(10))
print(sigmoid(100)) # 1에 수렴

xx,yy = [], []
for i in range(-100,101):
    n = sigmoid(i/10)

    xx.append(i/10)
    yy.append(n)

plt.plot(xx,yy,'ro')
plt.show()