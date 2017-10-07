#!/usr/bin/python3

''' In this example, we're going to use linear regression in tensorflow to predict housing prices based
    on the size of the lot as our features. 
''' 
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.mlab import PCA
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import tensorflow as tf
import sys
from sklearn import model_selection
from sklearn import preprocessing

np.set_printoptions(precision=3,suppress=True)

def normalize_features(dataset):
    mu = np.mean(dataset,axis=0)
    sigma = np.mean(dataset,axis=0)
    return (dataset - mu)/sigma

def pca(dataset):

    plt.scatter(dataset[:,0],dataset[:,1])
    plt.plot()
    plt.show()
    results = PCA(dataset)

    x = []
    y = []

    for item in results.Y:
        x.append(item[0])
        y.append(item[1])

    plt.close('all')
    fig1 = plt.figure()
    pltData = [x,y]
    plt.scatter(pltData[0],pltData[1],c='b')
    xAxisLine = ((min(pltData[0]),max(pltData[0])),(0,0),(0,0))
    yAxisLine = ((min(pltData[1]),max(pltData[1])),(0,0),(0,0))
    plt.xlabel('RM')
    plt.ylabel('MEDV')
    plt.show()
       

rng = np.random

# learning_rate is the alpha value that we pass to the gradient descent algorithm. 
learning_rate = 0.005


# How many cycles we're going to run to try and get our optimum fit. 
training_epochs = 1000
display_step =  50

# We're going to pull in a the csv file and extract the X value (RM) and Y value (MEDV)
boston_dataset = pd.read_csv('data/housing.csv')
label = boston_dataset['MEDV']
features = boston_dataset['RM'].reshape(-1,1)
features = normalize_features(features)
dataset = np.asarray(boston_dataset['RM'])
dataset = np.column_stack((np.asarray(boston_dataset['RM']),np.asarray(boston_dataset['MEDV'])))
pca(dataset)


train_X, test_X, train_Y, test_Y = model_selection.train_test_split(features, label, test_size = 0.33, random_state = 5)


scaler =  preprocessing.StandardScaler()
train_X = scaler.fit_transform(train_X)

# This is the total number of data samples that we're going to run through. 
n_samples = train_X.shape[0]

# Variable placeholders. 
X = tf.placeholder('float')
Y = tf.placeholder('float')

W = tf.Variable(rng.randn(), name = 'weight')
b = tf.Variable(rng.randn(), name = 'bias')
R_squared = tf.Variable(0,name='R_squared')


# Here we describe our training model.  It's a linear regression model using the standard y = mx + b 
# point slope formula. We calculate the cost by using least mean squares.

# This is our prediction algorithm: y = mx + b
prediction = tf.add(tf.multiply(X,W),b)

# Let's now calculate the cost of the prediction algorithm using least mean squares
training_cost = tf.reduce_sum(tf.pow(prediction-Y,2))/(2 * n_samples)
unexplained_cost = tf.reduce_sum(tf.square(tf.subtract(Y,prediction)))
R_squared =  tf.subtract(1.0, tf.divide(training_cost,unexplained_cost))

# This is our gradient descent optimizer algorithm.  We're passing in alpha, our learning rate
# and we want the minimum value of the training cost.  
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(training_cost)

init = tf.global_variables_initializer()

# Now we'll run our training data through our model.
with tf.Session() as tf_session:

# Initialize all of our tensorflow variables.
    tf_session.run(init)

# We'll run the data through for 1000 times (The value of training_epochs). 

    for epoch in range(training_epochs):

        # For each training cycle, pass in the x and y values to our optimizer algorithm to calculate the cost.
        for (x,y) in zip(train_X,train_Y):
            tf_session.run(optimizer,feed_dict = {X: x, Y: y})

        # For every fifty cycles, let's check and see how we're doing. 
        if (epoch + 1 ) % 50 == 0:
            c = tf_session.run(training_cost,feed_dict = {X: train_X, Y: train_Y})
            print ('Epoch: ', '%04d' % (epoch+1),'cost=','{:.9f}'.format(c), \
                   'W = ',tf_session.run(W), 'b = ',tf_session.run(b))


    print ('Optimization finished')
    print ('Training cost = ',training_cost,' W = ',tf_session.run(W), ' b  = ', tf_session.run(b),'\n')
    print ('R squared = ', tf_session.run(R_squared,feed_dict={X: train_X,Y: train_Y}))

    plt.plot(train_X, train_Y, 'ro',label='Original data')
    
    plt.plot(train_X,tf_session.run(W) * train_X + tf_session.run(b), label = 'Fitted line')
    plt.legend()
    plt.show()
    
    # We're now going to run test data to see how well our trained model works. 

    print ('Testing...(mean square loss comparison)')
    testing_cost = tf_session.run(tf.reduce_sum(tf.pow(prediction - Y, 2)) / (2 * test_Y.shape[0]),
                              feed_dict = {X: test_X, Y: test_Y})
    print ('Testing cost = ',testing_cost)
    print ('Absolute mean square loss difference: ', abs(training_cost  - testing_cost))

    plt.plot(test_X,test_Y,'bo',label='Testing data')
    plt.plot(test_X,tf_session.run(W) * test_X + tf_session.run(b), label = 'Fitted line')
    plt.legend()
    plt.show()



