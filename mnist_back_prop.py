# mnist_back_prop.py

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import random
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets('Data/mnist/',one_hot=True)

X = tf.placeholder(tf.float32,shape=[None,784])
Y = tf.placeholder(tf.float32,shape=[None,10])

W1 = tf.Variable(tf.random_normal([784,30]), name='weight1')
b1 = tf.Variable(tf.random_normal([30]), name='bias1')
W2 = tf.Variable(tf.random_normal([30,10]), name='weight2')
b2 = tf.Variable(tf.random_normal([10]), name='bias2')

# def sigmoid(z):
#     return 1./(1. + math.e**-z)
def sigma(x): # sigmoid function
    return tf.div(tf.constant(1.0),
                  tf.add(tf.constant(1.0), tf.exp(-x)))

def sigma_prime(x) : # derivate of the sigmoid function
    return sigma(x)*(1-sigma(x))

# Forward prop
l1 = tf.add(tf.matmul(X,W1),b1)
a1 = sigma(l1)
l2 = tf.add(tf.matmul(a1,W2),b2)
y_pred = sigma(l2)

# Backward prop(chain rule)
# diff
diff = (y_pred - Y)

d_l2 = diff*sigma_prime(l2)
d_b2 = d_l2
d_w2 = tf.matmul(tf.transpose(a1),d_l2)

d_a1 = tf.matmul(d_l2,tf.transpose(W2))
d_l1 = d_a1*sigma_prime(l1)
d_b1 = d_l1
d_w1 = tf.matmul(tf.transpose(X),d_l1)

# update weight and bias (network) : using gradients
learning_rate = 0.5
step = [
    tf.assign(W1, W1 - learning_rate*d_w1 ),
    tf.assign(b1, b1 - learning_rate*
              tf.reduce_mean(d_b1, reduction_indices=[0])),
    tf.assign(W2, W2 - learning_rate*d_w2 ),
    tf.assign(b2, b2 - learning_rate*
              tf.reduce_mean(d_b2, reduction_indices=[0]))
]
# Running
acct_mat = tf.equal(tf.argmax(y_pred,1),tf.argmax(Y,1))
acct_res = tf.reduce_mean(tf.cast(acct_mat,tf.float32))
acct_res = tf.reduce_mean(tf.cast(acct_mat,tf.float32))

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

for i in range(10000):
    batch_xs, batch_ys = mnist.train.next_batch(10)
    sess.run(step, feed_dict = { X:batch_xs, Y:batch_ys})

    if i % 1000 == 0 :
        res = sess.run(acct_res, feed_dict= {
            X:mnist.test.images[:1000],
            Y:mnist.test.labels[:1000] })
        print(res)

# Automatic in Tensoflow
cost = diff*diff
step = tf.train.GradientDescentOptimizer(cost)
