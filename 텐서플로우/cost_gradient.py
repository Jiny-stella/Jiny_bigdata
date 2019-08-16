# cost_gradient.py

import matplotlib.pyplot as plt

def cost(x,y,w):
    c = 0
    for i in range(len(x)):
        hx = w * x[i]
        loss = (hx - y[i])**2
        c += loss
    return c / len(x)

def call_cost():
    x = [1,2,3]
    y = [1,2,3]
    #y = [2,4,7]

    print(cost(x,y,-1))
    print(cost(x,y,0))
    print(cost(x,y,1))
    print(cost(x,y,2))
    print(cost(x,y,3))

    for i in range(-30,50):
        w = i/10
        c = cost(x,y,w)

        print(w,c)
        plt.plot(w,c,'ro')

    plt.show()

# 미분 : 순간 변화량, 기울기
# x축으로 1만큼 움직였을 때 y축으로 움직인 거리
# y = 3  ---> y' = 0
# y = 2x ----> y'= 2
# y = x^2 ----> y' = 2x
# y = (x + 1)^2 ----> y' = 2(x + 1)

def gradient_descent(x,y,w):
    grad = 0
    for i in range(len(x)):
        hx = w * x[i]
        grad += (hx - y[i])*x[i]
    return grad / len(x)


x = [1,2,3]
y = [1,2,3]
#y = [2,4,7]

w = 10
old = 100
for i in range(100):
    c = cost(x,y,w)
    grad = gradient_descent(x,y,w)
    w = w - 0.1*grad  # Learing Rate : 0.1
    print(i,c,w,grad)
    if c >= old and abs(c - old) < 1.0e-15:
        break
    old = c
    plt.plot((0,5),(0,5*w))

print('weight =',w)

plt.plot(x,y,'ro')
plt.xlim(0,5)
plt.ylim(0,5)
plt.show()