# tensorflow_basic.py

import tensorflow as tf

hello = tf.constant("Hello, Tensorflow!")
print(hello)
sess = tf.Session()
result = sess.run(hello)
print(result)

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
node3 = tf.add(node1,node2)

print("node1:", node1)
print("node2:", node2)
print("node3:", node3)

sess = tf.Session()

#print("sess.run([node1,node2]):", sess.run([node1,node2]))
print("sess.run(node3):", sess.run(node3))


a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = tf.add(a, b)

sess = tf.Session()
print(sess.run(adder_node, feed_dict={a:3, b:4.5}))
print(sess.run(adder_node, feed_dict={a:[1,3,5,7], b:[2,4,6,8]}))

a1 = tf.Variable(4.0,tf.float32)
b1 = tf.Variable(7.5,tf.float32)
adder_node2 = tf.add(a1, b1)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(adder_node2))