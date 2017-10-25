#!/usr/bin/env python3

import tensorflow as tf
import os
import sys



LOGDIR = 'tmp/mnist_tutorial/`'
LABELS = os.path.join(os.getcwd(), 'labels_1024.tsv')
SPRITES = os.path.join(os.getcwd(), 'sprite_1024.png')
print ('Loading data')
mnist = tf.contrib.learn.datasets.mnist.read_data_sets(train_dir = LOGDIR + 'data', one_hot = True)
if not (os.path.isfile(LABELS) and os.path.isfile(SPRITES)):
    print ('Necessary data files not found')
    sys.exit(1)

def conv_layer(input, channels_in, channels_out):
    w = tf.Variable(tf.zeros([5,5,channels_in, channels_out]))
    b = tf.Variable(tf.zeros([channels_out]))
    conv = tf.nn.conv2d(input,w,strides = [1,1,1,1], padding ='SAME')
    act = tf.nn.relu(conv + b)
    return act

def fc_layer(input,channels_in,channels_out):
    w = tf.Variable(tf.zeros([channels_in, channels_out]))
    b = tf.Variable(tf.zeros([channels_out]))
    act = tf.nn.relu(tf.add(tf.matmul(input,w),b))
    return act

x = tf.placeholder(tf.float32,shape=[None,784])
y = tf.placeholder(tf.float32, shape = [None,10])
x_image =- tf.reshape(x,[-1,28,28,1])

conv1 = conv_layer(x_image,1,32)
pool1 = tf.nn.max_pool(conv1,ksize = [1,2,2,1],strides = [1,2,2,1], padding = 'SAME')

conv2 = conv_layer(pool1,32,64)
pool2 = tf.nn.max_pool(conv2,ksize = [1,2,2,1],strides = [1,2,2,1], padding = 'SAME')
flattened = tf.reshape(pool2,[-1,7 * 7 *  64])

fc1 = fc_layer(flattened,7 * 7 * 64,1024)
logits = fc_layer(fc1,1024,10)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels=y))

train_step = tf.train.AdamOptimizer(.00001).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:

# Step 1.  Add the FileWriter step to write the logs out to disk.
    writer = tf.summary.FileWriter('/tmp/mnist_demo/1')
    writer.add_graph(sess.graph)
    print ('Initializing graph variables')
    sess.run(tf.global_variables_initializer())
    for i in range(2000):
        batch = mnist.train.next_batch(100)
        if i % 100 == 0:
            (train_accuracy) = sess.run((accuracy), feed_dict = {x: batch[0], y: batch[1]})
            print ('Step %d Accuracy %g' % (i,train_accuracy))
        sess.run(train_step,feed_dict = {x: batch[0], y: batch[1]})
                    








