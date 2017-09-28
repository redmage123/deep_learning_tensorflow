#!/usr/bin/python3

'''  
   The input data will be weighted and sent to our hidden layer 1. 
   We'll run this data through an activation function to send output data
   and weights to hidden layer 2.   The hidden layer 2 will then send
   its outputs and weights to an output layer. 

   This is known as a feed forward neural network as the data is passed
   straight through from the input layer to the output layer. 

   InLayer -> (inputs, weights) -> HL1(activation function) -> 
   (outputs, weights) -> HL2 (activation function) -> (outputs,weights)
   OL


   We'll compare the actual output to the intended output using a
   cost function (cross entropy)

   We'll use an optimization function (optimizer) to minimize our cost. 


   back propagation will then go and recalibrate the weights. 

   feedforward + backprop = epoch. 

   Each epoch is a training cycle.

''' 


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data



# ten classes 0-9.  

''' 
   The output (using the one_hot param) will look like this:
   0 = [1,0,0,0,0,0,0,0,0,0,]
   1 = [0,1,0,0,0,0,0,0,0,0,]
   2 = [0,0,1,0,0,0,0,0,0,0,]
   3 = [0,0,0,1,0,0,0,0,0,0,]
   4 = [0,0,0,0,1,0,0,0,0,0,]
   5 = [0,0,0,0,0,1,0,0,0,0,]
   6 = [0,0,0,0,0,0,1,0,0,0,]
   7 = [0,0,0,0,0,0,0,1,0,0,]
   8 = [0,0,0,0,0,0,0,0,1,0,]
   9 = [0,0,0,0,0,0,0,0,0,1,]

   We'll use three hidden layers. 
   We'll use ten classes (each number is a class). 

   The batch size will be 100.  We'll go through a batch of 100
   of features at a time, then manipulate the weights.  This is value is
   set for memory constraint reasons.

'''

mnist = input_data.read_data_sets('/tmp/data',one_hot = True)
n_nodes_hl1 = 500
n_nodes_hl2  = 500
n_nodes_hl3 = 500

n_classes = 10
batch_size = 10

# We're squashing the shape of the data.  Instead of 28x 28 matrix, 
# we're going to flatten it out to a single dimensional vector. 
x = tf.placeholder('float'[None, 784])
y = tf.placeholder ('float')

def neural_network_model(data):
    ''' This is the function that will model our neural network
    '''

# Creating a tensor of weights using random numbers with the shape
# of the number of nodes multiplied by the number of inputs

# (input data * weights) + biases. 
# Biases can be used to make sure that the neural network will fire even if 
# all the input values are zeroes. 

    hidden_1_layer = {'weights': tf.Variable(tf.random.normal([784, n_nodes_hl1])),
                      'biases':tf.Variable(tf.random_normal(n_nodes_hl1))} 

    hidden_2_layer = {'weights': tf.Variable(tf.random.normal([n_nodes_hl1, n_nodes_hl12])),
                      'biases':tf.Variable(tf.random_normal(n_nodes_hl1))} 

    hidden_3_layer = {'weights': tf.Variable(tf.random.normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases':tf.Variable(tf.random_normal(n_nodes_hl1))} 

    output_layer = {'weights': tf.Variable(tf.random.normal([784, n_nodes_hl1])),
                      'biases':tf.Variable(tf.random_normal(n_classes))} 




    # Here we multiply the input data * weights and add the biases.
    # input_data * weights + biases. 
    layer_1 = tf.add(tf.matmul(data,hidden_1_layer['weights']) * hidden_1_layer['biases'])

    # This describes the activation function for the hidden layer 1 nodes. 
    layer_1 = tf.nn.relu(l1) 

    layer_2 = tf.add(tf.matmul(layer_1, hidden_2_layer['weights']) + hidden_2_layer['biases'])
    layer_2 = tf.nn.relu(layer_2) 
    layer_3 = tf.add(tf.matmul(layer_2, hidden_3_layer['weights']) + hidden_3_layer['biases'])
    layer_3 = tf.nn.relu(layer_3) 

    output = tf.add(tf.matmul(layer_3, output_layer['weights']) + output_layer['biases'])
    return output



