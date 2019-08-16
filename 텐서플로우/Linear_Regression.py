# Linear_Regression.py
# Using tensorflow

import tensorflow as tf
tf.set_random_seed(777)

# X and Y data
x_train = [1,2,3]
y_train = [1,2,3]
#y_train = [4,7,10]

W = tf.Variable(tf.random_normal([1]), name = 'weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')

hypothesis = x_train * W + b

cost = tf.reduce_mean(tf.square(hypothesis - y_train ))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# start learning
for step in range(2001):
    sess.run(train)
    if step % 20 == 0 :
        print(step,sess.run(cost),sess.run(W),sess.run(b))

