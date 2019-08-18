# Multi linear_regression
def not_used():
    x1_data = [73.,93.,89.,96.,73.]
    x2_data = [80.,88.,91.,98.,66.]
    x3_data = [75.,93.,90.,100.,70.]

    y_data = [152.,185.,180.,196.,142.]
    
    x1 = placeholder(tf.float32)
    x2 = placeholder(tf.float32)
    x3 = placeholder(tf.float32)
    
    y = tf.placeholder(tf.float32)
    
    w1 = tf.Variable(tf.random_normal([1]), name = 'weight1')
    w2 = tf.Variable(tf.random_normal([1]), name = 'weight2')
    w3 = tf.Variable(tf.random_normal([1]), name = 'weight3')
    b = tf.Variable(tf.random_normal([1]), name = 'bias')
    
    #왜 단일 hypothesis에서는 x_data *w 를 했쥥ㅇ이이이이이이이이이이잉?
    hypothesis = x1 *w1 + x2*w2 + x3*w3 +b
    
    cost = tf.reduce_mean(tf.square(hypothesis - y))
    
    #cost 최소화
    Optimizer = tf.train.GradientdDesentOptimizer(learning_rate = 0.05)
    train = Optimizer.minimize(cost)
    
    #초기화
    sess = tf.Session()
    sess.run(tf.global_Variable_initializer())
    
   #start training
   for step in range(1001):
      cost_val,w1_val,w2_val,w3_val,b_val = \
      sess.run([cost,w1,w2,w3,b,train] , 
        feed_dict ={x1:x1_data,x2:x2_data,x3:x3_data, y:y_data})
        
      if step % 20 == 0:
          pint(step, cost_val, w1_val, w2_val, w3_val, b_val)
          
     ---------------------------------------------------------------------------
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
          
x = tf.placeholder(tf.float32,shape=[None,3])
y = tf.placeholder(tf.float32, shape =[None,1])

w = tf.Variable(random_normal([3,1]), name = 'weight')
b = tf.Variable(random_normal([1]), name = 'bias')

hypothesis = tf.matmul(x,w) + b 
cost = tf.reduce_mean(tf.square(hypothesis - y))

Optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.05)
train = Optimizer.minimize(cost)

