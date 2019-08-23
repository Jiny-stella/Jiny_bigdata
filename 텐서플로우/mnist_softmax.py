# mnist_softmax.py
# simple layer

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import random
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets('Data/mnist/',one_hot=True)

# print(type(mnist))
# print(type(mnist.train))

learning_rate = 0.01
training_epochs = 15
batch_size = 100

X = tf.placeholder(tf.float32,shape=[None,784]) # 28*28
Y = tf.placeholder(tf.float32,shape=[None,10])

W = tf.Variable(tf.random_normal([784,10]), name='weight')
b = tf.Variable(tf.random_normal([10]), name='bias')

logits = tf.matmul(X,W) + b # (?, 784) * (784, 10) = (? 10)
hypothesis = tf.nn.softmax(logits)

#cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis), axis=1))

cost =  tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits,
                                             labels = Y))
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# start training
for epoch in range(training_epochs) :  # 15
    avg_cost = 0
    # 550 = 55000/100
    total_batch = int(mnist.train.num_examples/batch_size)
    for i in range(total_batch) :
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        feed_dict = { X:batch_xs , Y:batch_ys}
        c,_= sess.run([cost,optimizer],feed_dict = feed_dict )
        avg_cost += c / total_batch
    print('Epoch:','%04d'%(epoch + 1), 'cost:','{:.9f}'.format(avg_cost))

print("Learning Finished!!")

# Test model and check accuracy
# accuracy computation
predict = tf.argmax(hypothesis,1)
correct_predict = tf.equal(predict,tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_predict,
                                     dtype = tf.float32))
a = sess.run([accuracy],feed_dict={X:mnist.test.images ,Y:mnist.test.labels})
print('train images total number = ', mnist.train.num_examples) # 55000
print('test image total number = ', mnist.test.num_examples)    # 10000
print("\nAccuracy:",a)

# get one random test data and predict
r = random.randint(0,mnist.test.num_examples - 1) # 0 to 9999 random int number
print("random=",r, "Label:", sess.run(tf.argmax(mnist.test.labels[r:r+1],1 )))

print("Prediction :", sess.run(tf.argmax(hypothesis,1),
                               feed_dict = { X: mnist.test.images[r:r+1]} ))

# matplotlib : imshow()
plt.imshow(mnist.test.images[r:r+1].reshape(28,28),
           cmap='Greys', interpolation='nearest')  # 2차원 보간법

plt.show()

# Epoch: 0012 cost: 0.289025393
# Epoch: 0013 cost: 0.284547325
# Epoch: 0014 cost: 0.284057295
# Epoch: 0015 cost: 0.280366603
# Learning Finished!!
# train images total number =  55000
# test image total number =  10000
#
# Accuracy: [0.9164]
# random= 4668 Label: [2]
# Prediction : [2]
