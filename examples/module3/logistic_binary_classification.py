#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection
import sys

gender_df = pd.read_csv('data/binary_data.csv',dtype = {col: np.float32 for col in ['HEIGHT'] })
print (gender_df.info())

# Shuffle our data
gender_df = gender_df.sample(frac=1).reset_index(drop=True)

num_features = 1
num_classes = 1
# We'll go ahead and split the data set into training and testing parts.  
# 70 per cent will go to training, the rest to testing. 
train_x,test_x, train_y, test_y = model_selection.train_test_split(gender_df['HEIGHT'],gender_df['GENDER'],test_size = 0.3)

n_samples = train_x.shape[0]

# These will be the placeholders for the testing and training data
x = tf.placeholder('float',[None,num_features+1])
y = tf.placeholder('float',[None,num_classes])

# Variables for the weight and bias. 
W = tf.Variable(tf.zeros([num_features, num_classes]))
b = tf.Variable(tf.zeros([num_classes]))

x = tf.reshape(x,[-1, num_features])
train_x = train_x.values.reshape(-n_samples,num_classes)
print ('w = ',W)
print ('x = ',x)
print ('train_x = ', train_x.shape)
# This is our activation function to determine the probability
# of our gender based on height. 
#activation = tf.nn.sigmoid((W * x) + b)
activation = tf.nn.softmax(tf.add(tf.matmul(x,W), b))

# Set our alpha value for the optimizer. 
learning_rate = 0.001

# cross_entropy is our cost function. 
cross_entropy = tf.reduce_mean(-(y*tf.log(activation) + (1 - y) * tf.log(1-activation)))

# We'll use a standard gradient descent optimizer.  
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

# Now train our jodel. 
    for epoch in range(1000):
        _,l = sess.run([train_step, cross_entropy], feed_dict = {x: train_x, y:train_y})
        if epoch % 50 == 0:
            print ('loss = %f' %(l))

# Now let's see how our model performed on the test data. 
    correct = tf.equal(tf.argmax(activation,1), tf.argmax(y,1))
    accuracy = tf.reduce_mean(tf.cast(correct,'float'))
    print ('Accuracy: ', sess.run(accuracy,feed_dict = {x: test_x, y:test_y}))


