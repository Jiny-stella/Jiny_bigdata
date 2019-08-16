# softmax_zoo_multi_classification.py

import tensorflow as tf
import numpy as np

xy = np.loadtxt('data-04-zoo.csv',delimiter=',',dtype=np.float32)
x_data = xy[:,0:-1]  # [101,16]
y_data = xy[:,[-1]]  # [101,1]

print(x_data.shape, y_data.shape)

nb_classes =  7

X = tf.placeholder(tf.float32,shape=[None,16])
Y = tf.placeholder(tf.int32,shape=[None,1])

Y_one_hot = tf.one_hot(Y,nb_classes)  # [None,1,7]
print(Y_one_hot)
Y_one_hot = tf.reshape(Y_one_hot,[-1,nb_classes]) # [None,7]
print(Y_one_hot)

W = tf.Variable(tf.random_normal([16,nb_classes]), name='weight')
b = tf.Variable(tf.random_normal([nb_classes]), name='bias')

logits = tf.matmul(X,W) + b
hypothesis = tf.nn.softmax(logits)

cost_i = tf.nn.softmax_cross_entropy_with_logits(logits = logits,
                                             labels = Y_one_hot)
cost =  tf.reduce_mean(cost_i)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# accuracy computation
predict = tf.argmax(hypothesis,1)
correct_predict = tf.equal(predict,tf.argmax(Y_one_hot,1))
accuracy = tf.reduce_mean(tf.cast(correct_predict,
                                     dtype = tf.float32))

# start training
for step in range(2001):
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
pred = sess.run(predict, feed_dict = {X:x_data})

for p,y in zip(pred, y_data.flatten()):
    print("[{}] Prediction: {} / Real Y: {}".format(p == int(y), p, int(y)))

