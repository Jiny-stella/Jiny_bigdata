# softmax_iris_deep_xavier_dropout.py
# setosa : [1,0,0]      --> [0]
# versicolor : [0,1,0]  --> [1]
# virginica : [0,0,1]   --> [2]

import tensorflow as tf
import numpy as np
import random

xy = np.loadtxt('Data/iris_softmax.csv',delimiter=',',dtype=np.float32)
x_data = xy[:,1:-3]
y_data = xy[:,-3:]

print(x_data.shape, y_data.shape)

nb_classes =  3
learning_rate = 0.001

X = tf.placeholder(tf.float32,shape=[None,4])
Y = tf.placeholder(tf.int32,shape=[None,3])

keep_prob = tf.placeholder(tf.float32)

W1 = tf.get_variable("W1", shape = [4,100],
            initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([100]), name='bias1')
# L1 = tf.sigmoid(tf.matmul(X,W1) + b1 )
L1 = tf.nn.relu(tf.matmul(X,W1) + b1 )
L1 = tf.nn.dropout(L1,keep_prob=keep_prob )

W2 = tf.get_variable("W2", shape = [100,100],
            initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([100]), name='bias2')
# L2 = tf.sigmoid(tf.matmul(L1,W2) + b2 )
L2 = tf.nn.relu(tf.matmul(L1,W2) + b2 )
L2 = tf.nn.dropout(L2,keep_prob=keep_prob )

W3 = tf.get_variable("W3", shape = [100,100],
            initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([100]), name='bias3')
# L3 = tf.sigmoid(tf.matmul(L2,W3) + b3 )
L3 = tf.nn.relu(tf.matmul(L2,W3) + b3 )
L3 = tf.nn.dropout(L3,keep_prob=keep_prob )

W4 = tf.get_variable("W4", shape = [100,100],
            initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([100]), name='bias4')
# L4 = tf.sigmoid(tf.matmul(L3,W4) + b4 )
L4 = tf.nn.relu(tf.matmul(L3,W4) + b4 )
L4 = tf.nn.dropout(L4,keep_prob=keep_prob )

W5 = tf.get_variable("W5", shape = [100,nb_classes],
            initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([nb_classes]), name='bias5')

logits = tf.matmul(L4,W5) + b5
hypothesis = tf.nn.softmax(logits)

cost =  tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits,
                                             labels = Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())


# accuracy computation
predict = tf.argmax(hypothesis,1)
correct_predict = tf.equal(predict,tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_predict, dtype = tf.float32))

# start training
for step in range(10001):
    cost_val,  _ = \
        sess.run([cost, optimizer],
                 feed_dict={X:x_data, Y:y_data, keep_prob : 0.7})
    if step % 20 == 0:
        print(step, cost_val)

# Accuracy report
h,p,a = sess.run([hypothesis,predict,accuracy],
                 feed_dict={X: x_data,Y:y_data, keep_prob : 1})
print("\nHypothesis:",h, "\nPredict:",p,"\nAccuracy:",a)


# predict : test model
r = random.randint(0,len(xy) - 1 ) # 0 to 149 random int number
print('x_data[r]:',x_data[r])
print('x_data[r:r+1]:',x_data[r:r+1])
print("random=",r, "Label:", sess.run(tf.argmax(y_data[r:r+1],1 )))

print("Prediction :", sess.run(tf.argmax(hypothesis,1),
                               feed_dict = { X: x_data[r:r+1], keep_prob : 1} ))

#
# Predict: [1 2 2 0 0 1 2 1 1 1 0 1 1 2 2 2 0 2 1 1 1 2 0 1 0 0 2 1 0 2 1 2 2 2 2 1 1
#  2 1 2 0 0 0 0 0 1 1 2 2 2 0 1 1 2 1 1 1 0 2 1 2 0 2 0 2 2 1 2 1 0 2 0 0 0
#  0 0 0 2 1 1 0 2 2 2 0 2 0 1 2 2 1 1 2 0 2 1 1 1 0 1 0 0 2 1 2 0 2 2 0 0 0
#  1 2 1 1 1 2 0 1 1 0 0 0 1 0 0 0 2 0 2 2 2 1 0 2 1 1 1 0 2 1 0 2 0 0 0 2 0
#  1 1]
# Accuracy: 1.0
# x_data[r]: [5.6 3.  4.1 1.3]
# x_data[r:r+1]: [[5.6 3.  4.1 1.3]]
# random= 30 Label: [1]
# Prediction : [1]