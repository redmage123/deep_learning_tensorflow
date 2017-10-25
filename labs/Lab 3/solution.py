#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn import model_selection
import matplotlib.pyplot as plt
import seaborn as sns
import sys

sns.set(style='white')
sns.set(style='whitegrid',color_codes=True)



bank_data = pd.read_csv('data/bank.csv',header=0,delimiter = ';')
bank_data = bank_data.dropna()

bank_data.drop(bank_data.columns[[0,3,8,9,10,11,12,13]],axis=1,inplace=True)
data_set = pd.get_dummies(bank_data,columns = ['job','marital','default','housing','loan','poutcome'])
data_set.drop(data_set.columns[[14,27]],axis=1,inplace=True)
data_set_y = data_set['y']
data_set_y = data_set_y.replace(('yes','no'),(1.0,0.0))

data_set_X = data_set.drop(['y'],axis=1)
num_samples = data_set.shape[0]
num_features = data_set_X.shape[1]
num_labels = 1


X = tf.placeholder('float',[None,num_features])
y = tf.placeholder('float',[None,num_labels])

W = tf.Variable(tf.zeros([num_features,1]),dtype=tf.float32)
b = tf.Variable(tf.zeros([1]),dtype=tf.float32)

train_X,test_X,train_y,test_y = model_selection.train_test_split(data_set_X,data_set_y,random_state=0)
train_y = np.reshape(train_y,(-1,1))

prediction = tf.add(tf.matmul(X,W),b)
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = prediction,labels = y))
optimizer = tf.train.GradientDescentOptimizer(0.00001).minimize(cost)
num_epochs = 1000


print ('Shape of train_y is: ',train_y.shape)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(num_epochs):
        _,l = sess.run([optimizer,cost],feed_dict = {X: train_X, y: train_y})
        if epoch % 50 == 0:
            print ('loss = %f' % (l))






