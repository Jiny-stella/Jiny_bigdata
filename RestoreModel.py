# Day_08_05_RestoreModel.py
import tensorflow as tf

x = [[1, 1, 1, 1, 1, 1],
     [2, 3, 3, 5, 7, 2],
     [1, 2, 5, 5, 5, 5]]
y =  [0, 0, 0, 1, 1, 1]

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W = tf.Variable(tf.random_uniform([1, 3], -1, 1))

z = tf.matmul(W, X)
hypo = tf.div(1., 1.+tf.exp(-z))
cost = -tf.reduce_mean(Y*tf.log(hypo) + (1-Y)*tf.log(1-hypo))

rate = tf.Variable(0.1)
train= tf.train.GradientDescentOptimizer(rate).minimize(cost)

sess = tf.Session()

# 복구한다면, 호출하면 안됨!!
# sess.run(tf.global_variables_initializer())

# ------------------------------------------ #
saver = tf.train.Saver()
# saver.restore(sess, 'Data/Logistic/logistic-180')
latest = tf.train.latest_checkpoint('Data/Logistic')
print('=====', latest)

if latest != None:
    saver.restore(sess, latest)
else:
    sess.run(tf.global_variables_initializer())

# ------------------------------------------ #

for i in range(200):
    sess.run(train, feed_dict={X: x, Y: y})
    if i%20 == 0:
        print(sess.run(cost, feed_dict={X: x, Y: y}), sess.run(W))

        # ------------------------------------------ #
        saver.save(sess, 'Data/Logistic/logistic', global_step=i)
        # ------------------------------------------ #

values = [[1, 1, 1, 1],
          [3, 4, 4, 6],
          [1, 2, 5, 4]]
print(sess.run(hypo, feed_dict={X: values}) > 0.5)

sess.close()
