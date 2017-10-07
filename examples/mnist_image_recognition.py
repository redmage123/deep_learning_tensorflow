#!/usr/bin/python3
import sys

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
x = tf.placeholder('float',[None, 784])
y = tf.placeholder ('float')

def neural_network_model(data):
    ''' This is the function that will model our neural network
    '''

# Creating a tensor of weights using random numbers with the shape
# of the number of nodes multiplied by the number of inputs

# (input data * weights) + biases. 
# Biases can be used to make sure that the neural network will fire even if 
# all the input values are zeroes. 

    hidden_1_layer = {'weights': tf.Variable(tf.random_normal([784, n_nodes_hl1])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))} 

    hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))} 

    hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))} 

    output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                      'biases':tf.Variable(tf.random_normal([n_classes]))} 



    # Here we multiply the input data * weights and add the biases.
    # input_data * weights + biases. 
    layer_1 = tf.add(tf.matmul(data,hidden_1_layer['weights']),hidden_1_layer['biases'])

    # This describes the activation function for the hidden layer 1 nodes. 
    layer_1 = tf.nn.relu(layer_1) 

    layer_2 = tf.add(tf.matmul(layer_1, hidden_2_layer['weights']),hidden_2_layer['biases'])
    layer_2 = tf.nn.relu(layer_2) 
    layer_3 = tf.add(tf.matmul(layer_2, hidden_3_layer['weights']),hidden_3_layer['biases'])
    layer_3 = tf.nn.relu(layer_3) 

    output = tf.matmul(layer_3, output_layer['weights']) + output_layer['biases']
    return output

def train_neural_network(x):
    ''' This function will train our neural network
        The prediction will be the output of the neural_network_model(x)
        The cost is tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediction,labels-y))
        This will calculate the difference from the prediction to the known label. 
        We want to minimize the difference gbetween the prediction to the known label. 
        We'll use an optimizer to try and minimize out difference. 
        We'll use the AdamOptimizer algorithm (stochastic descending gradient)
        The epoch is the feed forward plus the back propagation. 
    '''

    prediction = neural_network_model(x)
    print (prediction)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediction,labels = y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    num_epochs = 10

    with tf.Session() as tf_session:
        tf_session.run(tf.global_variables_initializer())

        for epoch in range(num_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples / batch_size)):
                epoch_x,epoch_y = mnist.train.next_batch(batch_size)
                _,c = tf_session.run([optimizer,cost],feed_dict= {x:epoch_x,y:epoch_y})
                epoch_loss += c
            print ('Epoch', epoch, 'completed out of ',num_epochs,'Epoch loss: ',epoch_loss)

        correct = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct,'float'))
        print ('Accuracy: ',accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))


train_neural_network(x)
    


