#!/usr/bin/env python3
import numpy as np
import tensorflow as tf

# Method 1.  Converting a numpy array to a tensor. 
a = np.zeros((3,3))
ta = tf.convert_to_tensor(a)
with tf.Session() as sess:
     print (sess.run(ta))

# Method 2.  Using a placeholder. 
# Here we declare a placeholder as a 32 bit floating point value. 
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.multiply(input1, input2)
with tf.Session() as sess:
    print (sess.run([output], feed_dict = {input1:[7.], input2: [2.]}))

