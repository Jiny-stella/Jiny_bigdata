# cnn_basic.py

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

sess = tf.InteractiveSession()

def not_used():
    image = np.array([[[[1],[2],[3]],
                       [[4],[5],[6]],
                       [[7],[8],[9]]]], dtype = np.float32)
    print(image.shape) # (1, 3, 3, 1)
    plt.imshow(image.reshape(3,3), cmap = 'Greys')
    plt.show()
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""
    # conv2d
    # image: (1,3,3,1) , Filter : (2,2,1,1). stride : (1,1)
    # number of images : 1, 3 * 3 image, color :1
    # Padding : VALID
    print("image:\n",image)
    print(image.shape) # (1, 3, 3, 1)

    weight = tf.constant([[[[1.]],[[1.]]],
                          [[[1.]],[[1.]]]])
    print("weight.shape:",weight.shape)
    # weigth.shape : (2,2,1,1)
    # 2 * 2 image, color :1, filters : 1

    conv2d = tf.nn.conv2d(image,weight,strides=[1,1,1,1], padding='VALID')
    conv2d_img = conv2d.eval()
    print('conv2d_img.shape:',conv2d_img.shape) # (1, 2, 2, 1)

    # 시각화
    conv2d_img = np.swapaxes(conv2d_img,0,3)
    for i, one_image in enumerate(conv2d_img) :
        print(one_image.reshape(2,2))
        plt.subplot(1,2,i+1)
        plt.imshow(one_image.reshape(2,2), cmap='Greys')
    plt.show()

    """"""""""""""""""""""""""""""""""""""""""""""""""""""""
    # conv2d
    # image: (1,3,3,1) , Filter : (2,2,1,1). stride : (1,1)
    # number of images : 1, 3 * 3 image, color :1
    # Padding : SAME
    print("image:\n",image)
    print(image.shape) # (1, 3, 3, 1)

    weight = tf.constant([[[[1.]],[[1.]]],
                          [[[1.]],[[1.]]]])
    print("weight.shape:",weight.shape)
    # weigth.shape : (2,2,1,1)
    # 2 * 2 image, color :1, filters : 1

    conv2d = tf.nn.conv2d(image,weight,strides=[1,1,1,1], padding='SAME')
    conv2d_img = conv2d.eval()
    print('conv2d_img.shape:',conv2d_img.shape) # (1, 3, 3, 1)

    # 시각화
    conv2d_img = np.swapaxes(conv2d_img,0,3)
    for i, one_image in enumerate(conv2d_img) :
        print(one_image.reshape(3,3))
        plt.subplot(1,2,i+1)
        plt.imshow(one_image.reshape(3,3), cmap='Greys')
    plt.show()

    """"""""""""""""""""""""""""""""""""""""""""""""""""""""
    # conv2d  3 filters : (2,2,1,3)
    # image: (1,3,3,1)   ==>  ouput : (1,3,3,3)
    weight = tf.constant([[[[1.,10.,-1.]],[[1.,10.,-1.]]],
                          [[[1.,10.,-1.]],[[1.,10.,-1.]]]])
    print("weight.shape:",weight.shape)
    # weigth.shape : (2,2,1,3)
    # 2 * 2 image, color :1, filters : 3

    conv2d = tf.nn.conv2d(image,weight,strides=[1,1,1,1], padding='SAME')
    conv2d_img = conv2d.eval()
    print('conv2d_img.shape:',conv2d_img.shape) # (1, 3, 3, 3)

    # 시각화
    conv2d_img = np.swapaxes(conv2d_img,0,3)
    for i, one_image in enumerate(conv2d_img) :
        print(one_image.reshape(3,3))
        plt.subplot(1,3,i+1)
        plt.imshow(one_image.reshape(3,3), cmap='Greys')
    plt.show()

    """"""""""""""""""""""""""""""""""""""""""""""""""""""""
    # max pooling (1, 2, 2, 1) --> (1, 1, 1, 1)
    # padding : VALID
    image = np.array([[[[4],[3]],
                       [[2],[1]]]], dtype=np.float32)
    print(image.shape) # (1, 2, 2, 1)

    pool = tf.nn.max_pool(image,ksize=[1,2,2,1],
                          strides=[1,1,1,1],padding='VALID')
    print(pool.shape) # (1, 1, 1, 1)
    print(pool.eval()) # [[[[4.]]]]

    """"""""""""""""""""""""""""""""""""""""""""""""""""""""
    # max pooling  (1, 2, 2, 1) --> (1, 2, 2, 1)
    # padding : SAME
    image = np.array([[[[4],[3]],
                       [[2],[1]]]], dtype=np.float32)
    print(image.shape) # (1, 2, 2, 1)

    pool = tf.nn.max_pool(image,ksize=[1,2,2,1],
                          strides=[1,1,1,1],padding='SAME')
    print(pool.shape) # (1, 2, 2, 1)
    print(pool.eval())
    # [[[[4.]
    #    [3.]]
    #   [[2.]
    #    [1.]]]]

    # """"""""""""""""""""""""""""""""""""""""""""""""""""""""

# MNIST image loading
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# Check out https://www.tensorflow.org/get_started/mnist/beginners for
# more information about the mnist dataset

img = mnist.train.images[0].reshape(28,28)
plt.imshow(img, cmap='gray')
plt.show()

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# MNIST Convolution layer
sess = tf.InteractiveSession()

img = img.reshape(-1,28,28,1)
print("img.shape", img.shape)  # img.shape (1, 28, 28, 1)

W1 = tf.Variable(tf.random_normal([3, 3, 1, 5], stddev=0.01)) # 3*3,  color:1 ,filters:5

# Output size : (N-F)/stride + 1
#     (29 - 3)/2 + 1 = 13 + 1 = 14  ,  ==> (1, 14, 14, 5)
conv2d = tf.nn.conv2d(img, W1, strides=[1, 2, 2, 1], padding='SAME')

print(conv2d) # (1, 14, 14, 5)
sess.run(tf.global_variables_initializer())
conv2d_img = conv2d.eval()
conv2d_img = np.swapaxes(conv2d_img, 0, 3)
for i, one_img in enumerate(conv2d_img):
    plt.subplot(1,5,i+1), plt.imshow(one_img.reshape(14,14), cmap='gray')

plt.show()


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# MNIST Max pooling
# conv2d : (1, 14, 14, 5)
# Output size : (N-F)/stride + 1
#     (14 - 2)/2 + 1 = 6 + 1 = 7 ,  ==> (1, 7, 7, 5)
pool = tf.nn.max_pool(conv2d, ksize=[1, 2, 2, 1], strides=[
                        1, 2, 2, 1], padding='SAME')
print(pool) # shape=(1, 7, 7, 5)

sess.run(tf.global_variables_initializer())
pool_img = pool.eval()

pool_img = np.swapaxes(pool_img, 0, 3)
for i, one_img in enumerate(pool_img):
    plt.subplot(1,5,i+1), plt.imshow(one_img.reshape(7, 7), cmap='gray')

plt.show()