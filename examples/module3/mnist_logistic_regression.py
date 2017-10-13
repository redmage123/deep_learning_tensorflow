#!/usr/bin/env python3

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random
import sys

# MNIST data is part of TensorFlow, so TF kindly supplies us 
# methods to retrieve this data directly. 
mnist = input_data.read_data_sets('MNIST_data',one_hot = True)


def TRAIN_SIZE(num):
    ''' TRAIN_SIZE returns a training set with num rows 
    ''' 

    print ('Total Training Image in Dataset = ' + str(mnist.train.images.shape))
    print ('------------------------------------')
   
    # Load num training set examples into our feature training set. 
    x_train = mnist.train.images[:num,:]
    print ('x_train Examples Loaded = ' + str(x_train.shape))
    
    # Load num training set examples into our labels training set.
    y_train = mnist.train.labels[:num,:]
    print ('y_train Examples Loaded = ' + str(y_train.shape))

    return x_train,y_train
    
def TEST_SIZE(num):
    ''' TEST_SIZE returns a test set with num rows 
    ''' 

    print ('Total Test Image in Dataset = ' + str(mnist.test.images.shape))
    print ('------------------------------------')
   
    # Load num training set examples into our feature training set. 
    x_test = mnist.test.images[:num,:]
    
    # Load num training set examples into our labels training set.
    y_test = mnist.test.labels[:num,:]

    return x_test,y_test

def display_digit(num):
    ''' This function will display a given numeric image (num) from mnist
        using matplotlib. 
    '''

    print (y_train[num])
    label = y_train[num].argmax(axis=0)
    image = x_train[num].reshape([28,28])
    plt.title('Example: %d Label: %d' % (num,label))
    plt.imshow(image,cmap=plt.get_cmap('gray_r'))
    plt.show()


def display_mult_flat(start,stop):
    images = x_train[start].reshape(1,784)
    for i in range(start+1,stop):
        images = np.concatenate((images,x_train[i].reshape((1,784))))
    plt.imshow(images,cmap=plt.get_cmap('gray_r'))
    plt.show()


# Let's start out by getting all of the images.  We'll run subsets of them
# to save computing resources. 
x_train,y_train = TRAIN_SIZE(55000)

#display_digit(random.randint(0,x_train.shape[0]))
#display_mult_flat(0,400)


# x is the placeholder in which we'll feed our train_x data. 
# Note that by setting x's dimensions to [None,784] we can 
# pass in as many 784 length pixel examples as we want
x = tf.placeholder(tf.float32, shape=[None,784])

# y is the placeholder for the label data. We'll use this
# to compare what we know to be true with what we
# predict to be true.  Note that we have ten 
# labels, corresponding to the values 0-9. 
y = tf.placeholder(tf.float32, shape=[None, 10])

# Now we want tod efine our weights and biases.  This is what 
# we want our classifier to tune so that we'll get accurate
# classification of our hand drawn digits. 
# We'll set our weights and biases to zero since TF will 
# optimize these values later on. Note that W is a 
# collection of 784 elements for each class. 
# b is a single value for each class. 

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

# Now let's feed our number of training examples fed in
# multiplied by the number of classes. 

# Note that softmax will take a set of values and force their sums
# to equal one.  This is what gives us our probabilities for 
# each value.   Softmax values will always be greater than zero
# and less than one. 
y_prob = tf.nn.softmax(tf.matmul(x,W) + b)

# Let's define the learning rate alpha for our gradient descent 
# optimizer. Remember, too large and it may not find our mininma.
# Too small, and it may take too long to find the value. 
learning_rate = 0.01

# Let's define the number of training steps to run
TRAIN_STEPS = 1000

# The cross_entropy function takes the log of all of our predictions y (Remember, 
# y values range from zero to one and does an element-wise multiplication by the
# examples true value.  if the log function for each value is close to zero, 
# then the output of that function will be a large negative value.  
# if the log function is close to one, it makes the value a very small 
# negative number.  We then take the negative of that to give us our 
# probability. 
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_prob), reduction_indices =[1]))

# We're going to use the standard GradientDescentOptimizer to find the
# minima of our cost function.  This will hopefully give us our best 
# combination of weight and bias to give us the most accurate
# classification. 

training = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_prob,1), tf.argmax(y_train,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Let's run the predictor for just three examples. 

#x_train,y_train = TRAIN_SIZE(3)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Note that the output of the print is going to be three probability vectors,
# one for each training example.  Each of the values in these vectors
# will be set to 0.1, which means that the softmax value is going to 
# calculate an equal 10 per cent chance that the image is a 0-9.
# print(sess.run(y,feed_dict={x:x_train}))

for i in range(TRAIN_STEPS + 1):
    sess.run(training, feed_dict = {x: x_train, y: y_train})
    if i % 50 == 0:
        l = sess.run(cross_entropy, feed_dict = {x: x_train, y:y_train})
        acc = sess.run(accuracy,feed_dict ={ x:x_train, y:y_train})
        print ('Training Step:',i, ' accuracy = ', acc,' loss = ' , l)



