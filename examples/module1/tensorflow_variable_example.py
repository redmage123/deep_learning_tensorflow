#!/usr/bin/env python3

import tensorflow as tf

W1 = tf.ones((2,2))

# Note that W2, unlike W1, is a node in the TensorFlow Graph. 
# The name parameter is useful when we want to save or restore
# the values of the variable from a file. 
W2 = tf.Variable(tf.zeros((2,2)),name='weights')

with tf.Session() as sess:
    print ('\nW1')
    print (sess.run(W1))

# Here we initialize all of the graph variaables.
    sess.run(tf.global_variables_initializer())

# We have to run the graph in order to gain access to W2 to print it.
    print ('\nW2')
    print (sess.run(W2))
