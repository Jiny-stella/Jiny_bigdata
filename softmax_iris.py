# softmax_iris.py
# setosa : [1,0,0]      --> [0]
# versicolor : [0,1,0]  --> [1]
# virginica : [0,0,1]   --> [2]

import tensorflow as tf
import numpy as np
import random

xy = np.loadtxt('iris_softmax.csv',delimiter=',',dtype=np.float32)
x_data = xy[:,1:-3]
y_data = xy[:,-3:]

print(x_data.shape, y_data.shape)

nb_classes =  3

X = tf.placeholder(tf.float32,shape=[None,4])
Y = tf.placeholder(tf.int32,shape=[None,3])

W = tf.Variable(tf.random_normal([4,nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')

logits = tf.matmul(X,W) + b
hypothesis = tf.nn.softmax(logits)

cost_i = tf.nn.softmax_cross_entropy_with_logits(logits = logits,
                                             labels = Y)
cost =  tf.reduce_mean(cost_i)

# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)
optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# accuracy computation
predict = tf.argmax(hypothesis,1)
correct_predict = tf.equal(predict,tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_predict, dtype = tf.float32))

# start training
for step in range(10001):
    cost_val, W_val, b_val, _ = \
        sess.run([cost, W, b, optimizer],
                 feed_dict={X:x_data, Y:y_data})
    if step % 20 == 0:
        print(step, cost_val, W_val, b_val)

# Accuracy report
h,p,a = sess.run([hypothesis,predict,accuracy],
                 feed_dict={X: x_data,Y:y_data})
print("\nHypothesis:",h, "\nPredict:",p,"\nAccuracy:",a)


# predict : test model
r = random.randint(0,len(xy) - 1 ) # 0 to 149 random int number
print('x_data[r]:',x_data[r])
print('x_data[r:r+1]:',x_data[r:r+1])
print("random=",r, "Label:", sess.run(tf.argmax(y_data[r:r+1],1 )))

print("Prediction :", sess.run(tf.argmax(hypothesis,1),
                               feed_dict = { X: x_data[r:r+1]} ))

# Predict: [1 2 2 0 0 1 2 1 1 1 0 1 1 2 2 2 0 2 1 1 1 2 0 1 0 0 2 1 0 1 1 2 2 2 2 1 1
#  2 1 2 0 0 0 0 0 1 1 2 2 2 0 1 1 2 1 1 1 0 2 1 2 0 2 0 2 2 1 2 1 0 2 0 0 0
#  0 0 0 2 1 1 0 2 2 2 0 2 0 1 2 2 1 1 2 0 2 1 1 1 0 1 0 0 2 1 2 0 2 2 0 0 0
#  1 2 1 1 1 2 0 2 1 0 0 0 1 0 0 0 2 0 2 2 2 1 0 2 1 1 1 0 2 1 0 2 0 0 0 2 0
#  1 1]
# Accuracy: 0.9866667
# random= 50 Label: [0]
# Prediction : [0]




