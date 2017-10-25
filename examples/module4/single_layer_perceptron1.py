#!/usr/bin/env python3
import tensorflow as tf

T,F = 1.,-1.
bias = 1

''' 
    Modeling the OR binary operation in a single layer perceptron
'''


# Our training set.  We give it the four possible inputs plus a bias. 
train_x = [
            [T,T,bias],
            [T,F,bias],
            [F,T,bias],
            [F,F,bias],
          ]

# These are the four possible outputs for the inputs of the OR function
# i.e. True and True is True, everything else is False
train_y = [ 
            [T],
            [T],
            [T],
            [F],
          ]


# Randomize our weights
W = tf.Variable(tf.random_normal([3,1]))

# Our placeholder for whatever input data we're going top pass in. 
x = tf.placeholder(tf.float32,shape = [None,3])

# Our plaeholder for the output data when we train it.
y = tf.placeholder(tf.float32,shape=[4,1])

# This is our heaviside step function.  It only returns a 1 or a 0 depending on what
# what's passed in as the x value. 
def activation_function(x):
    is_greater = tf.greater(x,0)
    is_greater = tf.to_float(is_greater)

# Make sure that our value isn't negative. 
    doubled = tf.multiply(is_greater,2)
    return tf.subtract(doubled,1)


# Set our activation function. 
output = activation_function(tf.matmul(x,W))

# As we train it, set the error value for each weight in  training iteration.
error = tf.subtract(y,output)

# Get the mean square error for all of the possible weights
mse = tf.reduce_mean(tf.square(error))


# Calculate our backprop delta. 
delta = tf.matmul(x,error,transpose_a = True)

# Now fix our weights according to what the delta value was.  
train = tf.assign(W,tf.add(W,delta))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

err,target = 1,0
epoch,max_epochs = 0,10


while err> target and epoch < max_epochs:
    epoch +=1
    o,err,_ = sess.run([output,mse,train],feed_dict = {x:train_x,y:train_y})
    print ('epoch:',epoch, 'error: ',err)
    print ('output = ',o)
    


def my_or(a,b):
    if a not in [True,False] or b not in [True,False]:
        raise InvalidArgumentException

    if a is True:
        a = T
    else:
        a = F

    if b is True:
        b = T
    else:
        b = F

    test_x = [[a,b,1]]
    o = sess.run(output,feed_dict = {x: test_x,y:train_y})
    if o == 1.0:
        return True
    else:
        return False

print (my_or(True,True))
print (my_or(True,False))
print (my_or(False,True))
print (my_or(False,False))
  
