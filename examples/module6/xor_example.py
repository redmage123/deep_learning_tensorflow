#!/usr/bin/env python3

import tensorflow as tf
import numpy as np

from tflearn import DNN
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

''' We're going to train a relatively simple DNN to calculate the XOR
    boolean operation.
'''

# This is the set of possible values to feed to XOR
x_train = [[0,0],
           [0,1],
           [1,0],
           [1,1]
          ]

# Given the values fed, this is the expected output
y_train = [[0],[1],[1],[0]] 

# Let's define our DNN.  We've got one input layer with a single node, 
# one hidden layer with two nodes, and one output layer with a single node..  
input_layer = input_data(shape = [None,2])
hidden_layer = fully_connected(input_layer,2,activation = 'tanh') 
output_layer = fully_connected(hidden_layer,1,activation = 'tanh')



# Let's define our regression activation function for output layer. 
# we define the optimizing function to be stochastic gradient descent
# we define our loss function as binary cross entropy.  We define
# our learning rate to be 5. 
regression = regression(output_layer,optimizer = 'sgd',loss='binary_crossentropy',learning_rate = 1)
model = DNN(regression)

# Now we train the model. 
model.fit(x_train,y_train,n_epoch=5000,show_metric=True)


# Let's see how it worked. 
print ('Expected: ',[i[0] > 0 for i in y_train])
print ('Predicted: ',[i[0] > 0 for i in model.predict(x_train)])
