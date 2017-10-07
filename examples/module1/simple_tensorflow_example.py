#!/usr/bin/env python3

import tensorflow as tf

# Running in interactive mode is good for debugging as it 
# executes the tensorflow code straight away rather
# than having to create a session variable.
tf.InteractiveSession()

# Create our 'a' array as a 2 X 2 tensor initialized to zeros.
a = tf.zeros((2,2))

print (a)

# Create our 'b' array as a 2 X 2 tensor initialized to ones. 
b = tf.ones((2,2))

# Sum array a and b, reduction_indices is similar to 'axis'.  
# note that we have to invoke the eval() method to 
# actually get the reduce_sum method to run.
print (tf.reduce_sum(b,reduction_indices=1).eval())

# Print the shape of tensor 'a'.
print (a.get_shape())

# Reshape 'a' as a 1 X 4 tensor and print it out. 
print (tf.reshape(a,(1,4)).eval())
