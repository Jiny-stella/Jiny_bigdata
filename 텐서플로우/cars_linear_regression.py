# cars_linear_regression.py
# X:speed,Y:dist

# Car 의 속도(X) 와 제동거리 (Y)
# 데이터 셋 가져오기
import numpy as np
import tensorflow as tf
import  matplotlib.pyplot as plt

# delimiter: 구분자,
# unpack = True :  행과 열을 transpose 하여 읽어옴
# skiprows : 헤더를 제거하고 읽어옴

xy = np.loadtxt('cars.csv',unpack=True, delimiter=',',skiprows=1)
x = xy[0]
y = xy[1]

# train using placeholder

W = tf.Variable(tf.random_normal([1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

X = tf.placeholder(tf.float32, shape=[None])
Y = tf.placeholder(tf.float32, shape=[None])

hypothesis = X * W + b

cost = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)   # 학습을 미세조정 , 답을 nan으로 나올때는 학습률을 고쳐라
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# start training
for step in range(40001):
    cost_val, W_val, b_val, _ = \
        sess.run([cost, W, b, train],
                 feed_dict={ X:x, Y:y})
    if step % 20 == 0:
        print(step, cost_val, W_val, b_val)

# predict : test model
# 속도가 30과 50일 때 제동거리를 예측해보세요
print(sess.run(hypothesis, feed_dict = {X:[30, 50]}))
print(sess.run(hypothesis, feed_dict = {X:[10,11,12,24,25]}))

print(W_val, b_val)

# 시각화 : matplotlib사용
def prediction(x,W,b):
    return W*x + b

plt.plot(x,y,'ro')
plt.plot((0,25),(0,prediction(25,W_val,b_val)))
plt.plot((0,25),(prediction(0,W_val,b_val),prediction(25,W_val,b_val)))
plt.show()

