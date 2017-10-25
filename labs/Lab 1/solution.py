#!/usr/bin/env python3


import tensorflow as tf

## Part 1

TF_LOGDIR = '/tmp/lab1'
with tf.name_scope('Part_1'):
    c = tf.Variable(0,dtype= tf.float32,name = 'c')
    c = tf.assign(c,tf.add(tf.sqrt(8.),3.))

## Part 2
with tf.name_scope('Part_2'):
    x = tf.Variable(1,dtype=tf.float32,name = 'c')
    y = tf.Variable(1,dtype=tf.float32,name = 'c')

    r1 = tf.assign(y,tf.divide(y,2))
    r2 = tf.assign(x,tf.add(r1,x))

with tf.Session() as sess:

    writer = tf.summary.FileWriter(TF_LOGDIR)
    writer.add_graph(sess.graph)

    sess.run(tf.global_variables_initializer())
    print ('Part 1. ')
    print (c.eval())

    print ('Part 2.')
    print (x.eval())
    for i in range(20):
        r1.eval()
        r2.eval()
        print (y.eval())


