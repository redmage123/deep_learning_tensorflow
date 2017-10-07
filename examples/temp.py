#!/usr/bin/python3

''' In this example, we're going to expand our linear regression example in tensorflow to predict housing prices based
    on the size of the lot, and the age of  the house was built  as our features. 
''' 
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.mlab import PCA
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import tensorflow as tf
import datetime
import sys


# Normalize all of the features so that they're on the same numeric scale.
# Not doing this can lead to errors in the training process.
def normalize_features(dataset):
    mu = np.mean(dataset,axis=0)
    sigma = np.std(dataset,axis=0)
    return (dataset - mu)/ sigma

def age(year):
    return datetime.datetime.now().year - year

def div(val):
    return val/1000

def pca(data):
    print ("In PCA")
    results = PCA(data)
    print (results)

    x = []
    y = []
    z = []

    for item in results.Y:
        x.append(item[0])
        y.append(item[1])
        z.append(item[2])
    plt.close('all')
    fig1 = plt.figure()
    ax = Axes3D(fig1)
    pltData = [x,y,z]
    ax.scatter(pltData[0],pltData[1],pltData[2],'bo')
    xAxisLine = ((min(pltData[0]),max(pltData[0])),(0,0),(0,0))
    yAxisLine = ((min(pltData[1]),max(pltData[1])),(0,0),(0,0))
    zAxisLine = ((min(pltData[2]),max(pltData[2])),(0,0),(0,0))

    ax.set_xlabel('Lot Size')
    ax.set_ylabel('Age')
    ax.set_zlabel('Sale Price')
    ax.set_title('PCA analysis')
    plt.show()


rng = np.random

# learning_rate is the alpha value that we pass to the gradient descent algorithm. 
learning_rate = 0.01


# How many cycles we're going to run to try and get our optimum fit. 
training_epochs = 1000
display_step =  50

# We're going to pull in a the csv file and extract the X values (LotArea,YearBuilt) and Y value (SalesPrice)
data_df = pd.read_csv('data/data_train.csv')
training_dataset = data_df[['LotArea','YearBuilt','SalePrice']]
print ("Doing PCA")
pca(training_dataset)
sys.exit(1)


# Convert the year the house was built into the age of the house from the current year. 
training_dataset = training_dataset['YearBuilt'].apply(age)
training_dataset = training_dataset['LotArea'].apply(div)
training_dataset = training_dataset['SalePrice'].apply(div)

# We're going to use the house age as our feature, so we'll have to modify the 
# We're going to do some feature scaling here and divide by 1000 for the Lot Area and Sale Price 

train_X = [training_dataset['LotArea'].values[:40] ,training_dataset['YearBuilt']]
train_Y = training_dataset['SalePrice'].values[:40]
#train_X = normalize_features(training_dataset['LotArea'].values[:40] )
#train_Y = normalize_features(training_dataset['SalePrice'].values[:40] )
sys.exit(1)


#train_X = normalize_features(train_X)
#train_Y = normalize_features(train_Y)

#print (training_dataset['LotArea'].values[:40])
#print (training_dataset['SalePrice'].values[:40])
#print (train_X)
#print (train_Y)


# This is the total number of data samples that we're going to run through. 
n_samples = train_X.shape[0]

# Variable placeholders. 
X = tf.placeholder('float')
Y = tf.placeholder('float')

W = tf.Variable(rng.randn(), name = 'weight')
b = tf.Variable(rng.randn(), name = 'bias')

# Here we describe our training model.  It's a linear regression model using the standard y = mx + b 
# point slope formula. We calculate the cost by using least mean squares.

# This is our prediction algorithm: y = mx + b
prediction = tf.add(tf.multiply(X,W),b)

# Let's now calculate the cost of the prediction algorithm using least mean squares
training_cost = tf.reduce_sum(tf.pow(prediction-Y,2))/(2*n_samples)

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

    plt.plot(train_X, train_Y, 'ro',label='Original data')
    plt.axis((0,2,0,5))
    
    plt.plot(train_X,tf_session.run(W) * train_X + tf_session.run(b), label = 'Fitted line')
    plt.legend()
    plt.show()
    
    # We're now going to run test data to see how well our trained model works. 
    data_df = pd.read_csv('data/data_test.csv')
    testing_dataset = data_df[['LotArea','SalePrice']]

    test_X = testing_dataset['LotArea'].values/1000
    test_Y = testing_dataset['SalePrice'].values/1000
#    test_X = normalize_features(testing_dataset['LotArea'].values[:40])
#    test_Y = normalize_features(testing_dataset['SalePrice'].values[:40])

    print ('Testing...(mean square loss comparison)')
    testing_cost = tf_session.run(tf.reduce_sum(tf.pow(prediction - Y, 2)) / (2 * test_Y.shape[0]),
                              feed_dict = {X: test_X, Y: test_Y})
    print ('Testing cost = ',testing_cost)
    print ('Absolute mean square loss difference: ', abs(training_cost  - testing_cost))

    plt.plot(test_X,test_Y,'bo',label='Testing data')
    plt.axis((0,2,0,5))
    plt.plot(test_X,tf_session.run(W) * test_X + tf_session.run(b), label = 'Fitted line')
    plt.legend()
    plt.show()



