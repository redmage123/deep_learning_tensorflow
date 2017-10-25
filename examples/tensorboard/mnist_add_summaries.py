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

# Adding scoping and naming to the graph. Add summaries.

def conv_layer(input, channels_in, channels_out,name='conv'):
    with tf.name_scope(name):
        w = tf.Variable(tf.zeros([5,5,channels_in, channels_out]),name = 'w')
        b = tf.Variable(tf.zeros([channels_out]),name = 'b')
        conv = tf.nn.conv2d(input,w,strides = [1,1,1,1], padding ='SAME')
        act = tf.nn.relu(conv + b)
        tf.summary.histogram('weights',w)
        tf.summary.histogram('biases',b)
        tf.summary.histogram('activation',act)
        return act

def fc_layer(input,channels_in,channels_out,name = 'fc'):
    with tf.name_scope(name):
        w = tf.Variable(tf.zeros([channels_in, channels_out]))
        b = tf.Variable(tf.zeros([channels_out]))
        act = tf.nn.relu(tf.add(tf.matmul(input,w),b))
        tf.summary.histogram('weights',w)
        tf.summary.histogram('biases',b)
        tf.summary.histogram('activation',act)
        return act

x = tf.placeholder(tf.float32,shape=[None,784],name='x')
x_image = tf.reshape(x,[-1,28,28,1])
tf.summary.image('input',x_image,3)
y = tf.placeholder(tf.float32, shape = [None,10],name='labels')

conv1 = conv_layer(x_image,1,32,'conv1')
pool1 = tf.nn.max_pool(conv1,ksize = [1,2,2,1],strides = [1,2,2,1], padding = 'SAME')

conv2 = conv_layer(pool1,32,64,'conv2')
pool2 = tf.nn.max_pool(conv2,ksize = [1,2,2,1],strides = [1,2,2,1], padding = 'SAME')
flattened = tf.reshape(pool2,[-1,7 * 7 *  64])

fc1 = fc_layer(flattened,7 * 7 * 64,1024,name='fc1')
logits = fc_layer(fc1,1024,10,name='fc2')

with tf.name_scope ('cross_entropy'):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels=y))
    tf.summary.scalar('cross_entropy',cross_entropy)

with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(.00001).minimize(cross_entropy)

with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy',accuracy)

with tf.Session() as sess:

# Step 1.  Add the FileWriter step to write the logs out to disk.
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter('/tmp/mnist_demo/3')
    writer.add_graph(sess.graph)
    print ('Initializing graph variables')
    sess.run(tf.global_variables_initializer())
    for i in range(2000):
        batch = mnist.train.next_batch(100)
        if i % 5 == 0:
            s = sess.run(merged_summary, feed_dict = {x: batch[0], y: batch[1]})
            writer.add_summary(s,i)

        if i % 100 == 0:
            (train_accuracy) = sess.run((accuracy), feed_dict = {x: batch[0], y: batch[1]})
            print ('Step %d Accuracy %g' % (i,train_accuracy))
        sess.run(train_step,feed_dict = {x: batch[0], y: batch[1]})
                    



