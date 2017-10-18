#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import scipy.io as io
#from matplotlib import pyplot as plt
#import plot_boundary_on_data

BATCH_SIZE = 100
out = np.loadtxt(filename,delimiters = ',')
labels = out[:,0]
features = out[:,1:]
num_features = features.shape

C = 1
num_epochs = 1


labels[labels == 0] = -1

x = tf.placeholder('float',shape=[None,num_features])
y = tf.placeholder('float',shape=[None,1])

w = tf.Variable(tf.zeros([num_features,1]))
b = tf.Variable(tf.zeros([1]))
y_raw = tf.matmul(x,W) + b

regularization_loss = 0.5 * tf.reduce_sum(tf.square(W))
hinge_loss = tf.reduce_sum(tf.maximum(tf.zeros([BATCH_SIZE,1]), 1 - y * y_raw))
svm_loss = regularization_loss + C * hinge_loss
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(svm_loss)

predicted_class = tf.sign(y_raw)
correct_prediction = tf.equal(y,predicted_class)
accuracy = tf.reduce_mean(tf.cast(correct_prediction,'float'))

with tf.Session() as sess:
    sess.run(tf.global_variables_initialize())
    for step in range(num_epochs ( train_size / BATCH_SIZE)):
        print ('Step = ',step)

        offset = (step * BATCH_SIZE) % train_size
        batch_data = train_data[offset:(offset + BATCH_SIZE),:]
        batch_labels = labels[offset:(offset + BATCH_SIZE),:]
        sess.run(train_step,feed_dict ={x: features, y: labels})
        print ('loss: ', svm_loss.eval(feed_dict = {x: features, y: labels}))



    









