#!/usr/bin/env python3

''' 
    input and weights and bias -> hidden layer 1 (activation functin)
    => hidden layer 2) -> repeat until we hit the output layer. 
    output layer -> activation function

    Compare intended to actual by using a cost function (cross entropy)
    Use an optimizer function -> minimize the cost (AdamOptimizer)

    Goes backwards to manipulate the weights -> back propagation

    feed_forward + back_propagation = epoch.
    each epoch (hopefully) lowers the value of the cost function, until it
    reaches a minimum. 

'''


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# Read the mnist data.  One_hot is for our logistic regression
# algrithm to take probabilities and turn them into a 0 or a 1. 
mnist = input_data.read_data_sets('/tmp/data',one_hot = True)


num_nodes_hl1 = 500
num_nodes_hl2 = 500
num_nodes_hl3 = 500

# How many classes does MNIST have?  (0-9)
num_classes = 10

# How many images at a time will we feed to the model?
batch_size = 100

# Define our placeholders for the features and labels
# We're flattening our 28X28 pixel array, we're going
# to unroll the matrix of pixels.  The first dimension
# of this matrix is None so that we can feed an arbitrary
# number of images into the placeholder. 
X = tf.placeholder('float',[None,784],name ='X')

# The y placeholder will hold the labels that we'll pass in. 
Y = tf.placeholder('float',name='Y')

# Let's define our neural network layers.  We'll define three hidden layers 
# and one output layer.  Note that for the weights,  it's the number of 
# input nodes for hidden layer 1.  For hidden layer 2 it's a matrix of the
# shape [number of nodes in hl1, number of nodes in hl2]
# For hidden layer 3 the shape is
# [number of nodes in hidden layer2,  number of nodes in hidden layer 3]
# Finally, for the output layer, the weights are the number of nodes in
# hidden layer 3 and the number of classes (the outputs). 
# 
# Note that we randomize the weights and biases using a normal (gaussian) distribution
# otherwise, our neural network won't learn anything. 

hidden_1_layer = {'weights':tf.Variable(tf.random_normal([784,num_nodes_hl1])),
                  'biases': tf.Variable(tf.random_normal([num_nodes_hl1]))}

hidden_2_layer = {'weights':tf.Variable(tf.random_normal([num_nodes_hl1,num_nodes_hl2])),
                  'biases': tf.Variable(tf.random_normal([num_nodes_hl2]))}


hidden_3_layer = {'weights':tf.Variable(tf.random_normal([num_nodes_hl2,num_nodes_hl3])),
                  'biases': tf.Variable(tf.random_normal([num_nodes_hl3]))}


output_layer = {'weights':tf.Variable(tf.random_normal([num_nodes_hl3,num_classes])),
                  'biases': tf.Variable(tf.random_normal([num_classes]))}


# Here's our summation for hidden layer 1.  We add up the matrix multiplication of the weights
# and the inputs, and add the biases. 
w1 = hidden_1_layer['weights']
b1 = hidden_1_layer['biases']

layer1_model =  tf.add(tf.matmul(X,w1), b1)

# Here's our activation function for each layer 1 node. 
# Note that we're using a Rectified Linear Unit (relu).  We could use sigmoid or softmax
# but relu neurons are computationally more efficient. It also approaches
# the optimization minima more quickly. However, we need to watch
# for dead relus!  Dead relu's happen where the weights are close to zero, which basically
# means that the relu will never (or almost never) fire. 

layer1_model = tf.nn.relu(layer1_model)


# Define the layer 2 model. 
w2 = hidden_2_layer['weights']
b2 = hidden_2_layer['biases']

layer2_model =  tf.add(tf.matmul(layer1_model, w2), b2)

# Here's our activation function for each layer 2 node. 
layer2_model = tf.nn.relu(layer2_model)

# Define the layer 3 model. 
w3 = hidden_3_layer['weights']
b3 = hidden_3_layer['biases']

layer3_model =  tf.add(tf.matmul(layer2_model, w3),b3)

# Here's our activation function for layer 3 node. 
layer3_model = tf.nn.relu(layer3_model)


ow = output_layer['weights']
ob = output_layer['biases']

# Our output layer sums up the product of the layer 3 weights and the output weights and the 
# output layer biases and gives us a result. 
prediction = tf.add(tf.matmul(layer3_model,ow),ob)

# Now let's figure out our error. 
#cost_ = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediction,labels = Y))
cost_func = tf.nn.softmax_cross_entropy_with_logits(logits = prediction,labels = Y)
cost = tf.reduce_mean(cost_func)

# This is our optimizer.  The defualt learning rate for it is 0.001
optimizer = tf.train.AdamOptimizer().minimize(cost)

# Let's define our many training epochs we want. 
num_epochs = 10

# We've built our computational graph, now let's run it. 

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(num_epochs):
        epoch_loss = 0
        for _ in range(int(mnist.train.num_examples/batch_size)):
            e_x,e_y = mnist.train.next_batch(batch_size)
            _,c = sess.run([optimizer,cost],feed_dict = {X:e_x,Y:e_y})
            epoch_loss += c
        print ('Epoch ', epoch, ' completed out of ', num_epochs,' loss: ',epoch_loss)


        correct = tf.equal(tf.argmax(prediction,1),tf.argmax(Y,1))
        accuracy = tf.reduce_mean(tf.cast(correct,'float'))

# Now let's run our predicted model through some mnist test data. 
        print ('Accuracy = ',accuracy.eval({X:mnist.test.images,Y:mnist.test.labels}))





