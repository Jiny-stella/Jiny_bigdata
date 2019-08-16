# logistic_regression.py

import tensorflow as tf

# x_data : [6,2]
x_data = [[1,2],
          [2,3],
          [3,1],
          [4,3],
          [5,3],
          [6,2]]

# y_data : [6,1]
y_data = [[0],
          [0],
          [0],
          [1],
          [1],
          [1]]

X = tf.placeholder(tf.float32,shape=[None,2])
Y = tf.placeholder(tf.float32,shape=[None,1])

W = tf.Variable(tf.random_normal([2,1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# sigmoid : tf.div(1., 1. + tf.exp(tf.matmul(X,W)))
hypothesis = tf.sigmoid(tf.matmul(X,W) + b )

cost = -tf.reduce_mean(Y*tf.log(hypothesis) + (1-Y)*
                       tf.log(1-hypothesis))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)
