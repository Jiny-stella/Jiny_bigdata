# multi_variable_linear_regression.py
# w : 입력값이 결과값에 영향을 주는 값, x의 갯수만큼 w 의 갯수가 필요하당

import tensorflow as tf

def not_used():
    x1_data = [73.,93.,89.,96.,73.]
    x2_data = [80.,88.,91.,98.,66.]
    x3_data = [75.,93.,90.,100.,70.]

    y_data = [152.,185.,180.,196.,142.]

    x1 = tf.placeholder(tf.float32)
    x2 = tf.placeholder(tf.float32)
    x3 = tf.placeholder(tf.float32)


    Y = tf.placeholder(tf.float32)

    W1 = tf.Variable(tf.random_normal([1]), name='weight1')  # 스칼라값으로 뽑는당 노말안에는 ([모양])
    W2 = tf.Variable(tf.random_normal([1]), name='weight2')
    W3 = tf.Variable(tf.random_normal([1]), name='weight3')
    b = tf.Variable(tf.random_normal([1]), name='bias')

    hypothesis = x1*W1 + x2*W2 + x3*W3 + b

    cost = tf.reduce_mean(tf.square(hypothesis - Y))

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
    train = optimizer.minimize(cost)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # start training
    for step in range(100001):
        cost_val, W1_val,W2_val,W3_val, b_val, _ = \   #역패킹, 역슬래시는 담줄 지금줄 연결
            sess.run([cost, W1, W2, W3, b, train],
                     feed_dict={ x1:x1_data,x2:x2_data,x3:x3_data, Y:y_data})
        if step % 20 == 0: # 이거는 % 나머지가 0인거어어어어어어?
            print(step, cost_val, W1_val,W2_val,W3_val, b_val)

    # predict : test model
    x1_data = [73.,93.,89.,96.,73.]
    x2_data = [80.,88.,91.,98.,66.]
    x3_data = [75.,93.,90.,100.,70.]

    print(sess.run(hypothesis, feed_dict = {x1:x1_data,x2:x2_data,x3:x3_data}))
    # y_data = [152.,185.,180.,196.,142.]
    # [150.79488 184.97044 180.28032 197.34216 141.10304] : 20001
    # [151.16731 184.78398 180.29572 196.88875 141.5592 ] : 40001
    # [151.45259 184.67514 180.81311 196.03458 141.92386] : 100001
# not_used()
# input()   # 인풋으로 멈추는건 .......................................................머지이이이이이이잉이이이이ㅣ

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# matrix  사용: tf.matmul() 사용
# x_data : [5,3]     # x1,x2,x3 데이터를 매트릭스 한번에 한거!!
x_data = [[73.,80.,75.],
          [93.,88.,93.],
          [89.,91.,90.],
          [96.,98.,100.],
          [73.,66.,70.]]
# y_data : [5,1]
y_data = [[152.],
          [185.],
          [180.],
          [196.],
          [142.]]

X = tf.placeholder(tf.float32,shape=[None,3])
Y = tf.placeholder(tf.float32,shape=[None,1])        #모양 중요!!!!!!!!!!! 근데 어케.....................?

W = tf.Variable(tf.random_normal([3,1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.matmul(X,W) + b  #x,w 순서 중요

cost = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# start training
for step in range(100001):
    cost_val, W_val, b_val, _ = \
        sess.run([cost, W, b, train],
                 feed_dict={X:x_data, Y:y_data})
    if step % 20 == 0:
        print(step, cost_val, W_val, b_val)


# predict : test model

print(sess.run(hypothesis, feed_dict = {X:x_data}))  # 앞에서 이미 w와 y 했기 때문에

#  [[151.44507]
#  [184.6806 ]
#  [180.83571]
#  [196.01993]
#  [141.9165 ]]


