#!/usr/bin/python3


""" Tensorflow is basically a library of array manipulation methods
    A tensor is just a one or a multidimensional array.
    tensorflow provides functions that work on tensors

    Tensorflow is really a deep learning library with many library
    routines.
    Most of tensorflow is written in C and C++ for performance.

    With tensorflow first you define your model in somewhat abstract terms,
    then you run it.  So it's stucturally different than a normal Python
    program.

""" 

import tensorflow as tf

# Let's construct our graph, X1 and X2 will be constants. 

X1 = tf.constant(1)
X2 = tf.constant(2)

# Obvious but inefficient way of multiplying two constants with
# Tensorflow
result = X1 * X2

# Better way. 

result = tf.multiply(X1,X2)

# Note that no multiplication will happen until we've created and run 
# the session. 

tfsession = tf.Session()
print (tfsession.run(result))
tfsession.close()


# A better way is to use the 'with' context object to do this. 
# with tf.Session() as tfsession:
# Note that output is a python variable, not a tensorflow session
# variable, so we can move values back and forth between normal 
# python and tensorflow session objects. 

    output = sess.run(result)
    print (output)

# Note that once you're outside the session, trying to run sess.run(result)
# will not work as the tensorflow session has been closed. 


