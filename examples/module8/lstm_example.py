#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
import random
import collections
import time
import sys
import pdb

def read_data(fname):
    with open(fname) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    content = [content[i].split() for i in range(len(content))]
    content = np.array(content)
    content = np.reshape(content,[-1,1])
    return content

def build_dataset(words):
    count = collections.Counter(np.ndarray.flatten(words)).most_common()
    word_dict = dict()
    for word, _ in count:
        word_dict[word] = len(word_dict)
    reverse_dict = dict(zip(word_dict.values(),word_dict.keys()))
    return word_dict,reverse_dict

def RNN(x,weights,biases):
    x = tf.reshape(x,[-1,n_input])
    x = tf.split(x,n_input,1)

    rnn_cell = rnn.BasicLSTMCell(n_hidden)
    outputs,states = rnn.static_rnn(rnn_cell,x,dtype=tf.float32)
    return tf.matmul(outputs[-1],weights['out']) + biases['out']


training_file = 'data/lstm_data.txt'
training_data = read_data(training_file)
w_dict,r_dict = build_dataset(training_data)
#print (w_dict)
#sys.exit()
vocab_size = len(w_dict)

learning_rate = 0.001
training_iters = 50000
display_step = 1000
n_input = 3

n_hidden = 512

x = tf.placeholder(tf.float32,[None,n_input,1])
y = tf.placeholder(tf.float32,[None,vocab_size])

weights = {'out': tf.Variable(tf.random_normal([n_hidden,vocab_size]))}
biases = {'out': tf.Variable(tf.random_normal([vocab_size]))}

pred = RNN(x,weights,biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = pred, labels=y))
optimizer =  tf.train.RMSPropOptimizer(learning_rate = learning_rate).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    step = 0
    offset = random.randint(0,n_input+1)
    end_offset = n_input + 1
    acc_total = 0
    loss_total = 0

#    writer.add_graph(sess.graph)

    print ('Training iters = ',training_iters)
    while step < training_iters:
        if offset > (len(training_data) - end_offset):
            offset = random.randint(0,n_input+1)

        symbols_in_keys = [[w_dict[str(training_data[i][0])]] for i in range(offset,offset+n_input)]
        symbols_in_keys = np.reshape(np.array(symbols_in_keys), [-1,n_input, 1])

        symbols_out_onehot = np.zeros([vocab_size],dtype=float)
        symbols_out_onehot[w_dict[str(training_data[offset+n_input][0])]] = 1.0
        symbols_out_onehot = np.reshape(symbols_out_onehot, [1,-1])

        _,acc,loss,onehot_pred = sess.run([optimizer, accuracy, cost, pred],feed_dict = {x:symbols_in_keys, y:symbols_out_onehot})

        loss_total += loss
        acc_total += acc
        if (step + 1) % display_step == 0:
            print ('Iter = ' + str(step+1) + ', Average Loss = ' + \
                   "{:.6f}".format(loss_total/display_step) + 'Average Accuracy = ' + \
                   "{:.2f}".format(100*acc_total/display_step))
            acc_total = 0
            loss_total = 0
            symbols_in = [training_data[i] for i in range(offset, offset + n_input)]
            symbols_out = training_data[offset + n_input]
            symbols_out_pred = r_dict[int(tf.argmax(onehot_pred, 1).eval())]
            print ('%s - [%s] vs [%s]' % (symbols_in, symbols_out, symbols_out_pred))
        step += 1
        offset += (n_input+1)


while True:
    prompt = "%s words: " % n_input
    sentence = input(prompt)
    sentence = sentence.strip()
    words = sentence.split(' ')
    if len(words) != n_input:
        continue
    try:
        symbols_in_keys = [w_dict[str(words[i])] for i in range(len(words))]
        for i in range(32):
            keys = np.reshape(np.array(symbols_in_keys), [-1, n_input, 1])
            onehot_pred = session.run(pred, feed_dict={x: keys})
            onehot_pred_index = int(tf.argmax(onehot_pred, 1).eval())
            sentence = "%s %s" % (sentence,reverse_dictionary[onehot_pred_index])
            symbols_in_keys = symbols_in_keys[1:]
            symbols_in_keys.append(onehot_pred_index)
    except:
        print("Word not in dictionary")

    print(sentence)
