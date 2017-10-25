#!/usr/bin/env python3

import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn import model_selection
import numpy as np
import pandas as pd
import sys

data_df = pd.read_csv('data/life_satisfaction.csv')
data_df.drop(data_df.columns[0], axis=1,inplace=True)

features = data_df['GDP per capita']
labels = data_df['Life satisfaction']
num_samples = features.shape[0]

x = tf.placeholder('float')
y = tf.placeholder('float')

w = tf.Variable(0,dtype=tf.float32)
b = tf.Variable(0,dtype=tf.float32)
train_x,test_x,train_y,test_y = modiel_selection.train_test_split(features,labels,random_state=0)

prediction = tf.add(tf.multiply(w,x),b)
cost = tf.reduce_sum(tf.pow(prediction - y,2))/(2 * num_samples)
optimizer = tf.train.GradientDescentOptimizer(0.0001).minimize(cost)


num_epochs = 1000
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(num_epochs):
        _,c = sess.run([optimizer,cost] feed_dict = {x:train_x, y:train_y})
        if epoch % 50 == 0:
            print ('Cost = %f ' % c)
    
