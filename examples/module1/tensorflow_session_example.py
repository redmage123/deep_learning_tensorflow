#!/usr/bin/env python3

import tensorflow as tf

# Here we declare two constants. 
a = tf.constant (5.0)
b = tf.constant (6.0)

# Multiply the two tensors together to output a third tensor 'c'.
c = a * b

# Nothing will run until we start a TensorFlow session.  
# In the previous example, tf.InteractiveSession() was just 
# syntactic sugar for the following. 
# Note that we can use the Python 'with' to treat tf.Session
# as a context object, containing __entry__ and __exit__ methods.
with tf.Session() as sess:

# Note that c.eval() is, again, syntactic sugar for the sess.run(c) 
# statment. sess.run() is an exxample of a Tensorflow fetch
# statement. 
    print (sess.run(c))
    print (c.eval())

