#!/usr/bin/env python3
import tensorflow as tf

import sys

class Perceptron: 

    T,F = 1.,-1.
    bias = 1
    train_x = [
                [T,T,bias],
                [T,F,bias],
                [F,T,bias],
                [F,F,bias],
              ]

    @staticmethod
    def activation_function(xval):
        is_greater = tf.greater(xval,0)
        is_greater = tf.to_float(is_greater)
        doubled = tf.multiply(is_greater,2)
        return tf.subtract(doubled,1)

    def __init__(self,y):
        print ('y = ',y)
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.train_y = y
            self.convert_bool_vals()
            self.W = tf.Variable(tf.random_normal([3,1]))
            self.x = tf.placeholder(tf.float32,shape = [None,3])
            self.y = tf.placeholder(tf.float32,shape=[4,1])
            self.output = Perceptron.activation_function(tf.matmul(self.x,self.W))
            self.error = tf.subtract(self.y,self.output)
            self.mse = tf.reduce_mean(tf.square(self.error))
            self.delta = tf.matmul(self.x,self.error,transpose_a = True)
            self.train = tf.assign(self.W, tf.add(self.W,self.delta))

    def convert_bool_vals(self):
        tmp = []
        for val in range(len(self.train_y)):
            if self.train_y[val] ==  True:
                self.train_y[val] = Perceptron.T
            else:
                self.train_y[val] = Perceptron.F
            tmp.append([self.train_y[val]])
        self.train_y = tmp
        

    def train_model(self):
        with tf.Session(graph=self.graph) as sess:
            sess.run(tf.global_variables_initializer())
            err,target = 1,0
            epoch,max_epochs = 0,10

            while err > target and epoch < max_epochs:
                epoch +=1
                err,t,d = sess.run([self.mse,self.train,self.delta],feed_dict = {self.x:Perceptron.train_x,self.y:self.train_y})
                print ('epoch:',epoch, 'error: ',err)
                print ('delta = ',d)
                print ('train = ',t)
                o = sess.run([self.output],feed_dict = {self.x:Perceptron.train_x,self.y:self.train_y})
                print ('o = ',o)

    def test_model(self,test_x):
        with tf.Session(graph = self.graph) as sess:
            sess.run(tf.global_variables_initializer())
            output = Perceptron.activation_function(tf.matmul(self.x,self.W))
            o = sess.run(output,feed_dict = {self.x: test_x, self.y:self.train_y})
            if o == 1.0:
                return True
            else:
                return False;



        
def main():
    AND_perceptron = Perceptron([[True],[False],[False],[False]])
    AND_perceptron.train_model()
    output = AND_perceptron.test_model([[1,1,1]])
    print ('output = ',output)
    output = AND_perceptron.test_model([[1,0,1]])
    print ('output = ',output)
    output = AND_perceptron.test_model([[0,1,1]])
    print ('output = ',output)
    output = AND_perceptron.test_model([[0,0,1]])
    print ('output = ',output)

    OR_perceptron = Perceptron([[True],[True],[True],[False]])
    OR_perceptron.train_model()
    print(OR_perceptron.test_model([[0.0,0.0,1]]))
    print(OR_perceptron.test_model([[1.0,0.0,1]]))
    print(OR_perceptron.test_model([[0.0,1.0,1]]))
    print(OR_perceptron.test_model([[1.0,1.0,1]]))

if __name__ == '__main__':
    main()
