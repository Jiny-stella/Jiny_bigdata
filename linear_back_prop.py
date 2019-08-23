# linear_back_prop.py
import tensorflow as tf

x_data = [[1.],
          [2.],
          [3.]]

y_data = [[1.],
          [2.],
          [3.]]

X = tf.placeholder(tf.float32,shape=[None,1])
Y = tf.placeholder(tf.float32,shape=[None,1])

W = tf.Variable(tf.random_normal([1,1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# forward prop
hypothesis = tf.matmul(X,W) + b

# diff
diff = (hypothesis - Y)

# backward prop
d_l1 = diff
d_b = d_l1
d_w = tf.matmul(tf.transpose(X),d_l1)

print(X,W,d_l1,d_w)

# updating network
learning_rate = 0.1
step = [
    tf.assign(W, W - learning_rate*d_w ),
    tf.assign(b, b - learning_rate*
              tf.reduce_mean(d_b)),
]

# Running
RMSE = tf.reduce_mean(tf.square((Y-hypothesis)))

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

for i in range(1000):

    print(i, sess.run([step,RMSE], feed_dict={X:x_data, Y:y_data}))

print(sess.run(hypothesis,feed_dict={X:x_data}))

# Automatic in Tensoflow
cost = diff*diff
step = tf.train.GradientDescentOptimizer(cost)


# 996 [[array([[1.0000004]], dtype=float32), array([-6.2468473e-07], dtype=float32)], 3.7895614e-14]
# 997 [[array([[1.0000004]], dtype=float32), array([-6.2468473e-07], dtype=float32)], 3.7895614e-14]
# 998 [[array([[1.0000004]], dtype=float32), array([-6.2468473e-07], dtype=float32)], 3.7895614e-14]
# 999 [[array([[1.0000004]], dtype=float32), array([-6.2468473e-07], dtype=float32)], 3.7895614e-14]
# [[0.99999976]
#  [2.        ]
#  [3.0000002 ]]
