
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 15:30:10 2017

@author: Shidong Li
"""
import tensorflow as tf
import tensorlayer as tl
import time
ISOTIMEFORMAT='%Y-%m-%d-%X'
# Data loading and preprocessing
#import data_process as data
#from data_process import nummfcc
#from data_process import N_frames
#X,Y,TestX,TestY= data.mfccset()
#X = X.reshape([-1, N_frames, nummfcc, 1])
#TestX = TestX.reshape([-1, N_frames, nummfcc, 1])
#  Use load data from npy instead
## Building convolutional network
data = tl.files.load_npy_to_any(name = 'data.npy')
X = data['X']
Y = data['Y']
TestX= data['TestX']
TestY= data['TestY']
N_frames= data['N_frames']
nummfcc= data['nummfcc']
sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32,shape= [None,N_frames,nummfcc,1])
digits = tf.placeholder(tf.int64,shape=[None,])
 
network = tl.layers.InputLayer(x,name = 'MfccGraphInput')
network = tl.layers.Conv2d(network, n_filter=32, filter_size=(5, 5), strides=(1, 1),
            act=tf.nn.relu, padding='SAME', name='cnn1')
network = tl.layers.MaxPool2d(network, filter_size=(2, 2), strides=(2, 2),
            padding='SAME', name='pool1')
network = tl.layers.BatchNormLayer(network,name = 'BNLayer1')
network = tl.layers.Conv2d(network, n_filter=64, filter_size=(5, 5), strides=(1, 1),
            act=tf.nn.relu, padding='SAME', name='cnn2')
network = tl.layers.MaxPool2d(network, filter_size=(2, 2), strides=(2, 2),
            padding='SAME', name='pool2')
network = tl.layers.BatchNormLayer(network,name = 'BNLayer2')
## end of conv
network = tl.layers.FlattenLayer(network, name='flatten')
network = tl.layers.DenseLayer(network, n_units=256,act = tf.nn.relu, name='relu1')
network = tl.layers.BatchNormLayer(network,name = 'BNLayer3')
network = tl.layers.DenseLayer(network, n_units=64,act = tf.nn.relu, name='relu2')
#network = tl.layers.DropoutLayer(network, keep=0.85, name='drop2')
network = tl.layers.BatchNormLayer(network,name = 'BNLayer4')
network = tl.layers.DenseLayer(network, n_units=10,act = tf.identity,name='output')
out = network.outputs
cost = tl.cost.cross_entropy(out,digits,name = 'cost')
Acc = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(out,1),digits), tf.float32))
train_params = network.all_params
train_op = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.9, beta2=0.999,
                            epsilon=1e-08, use_locking=False).minimize(cost, var_list=train_params)

# Initialize all variables in the session
tl.layers.initialize_global_variables(sess)
# Print network information
network.print_params()
network.print_layers()

# Train the network, we recommend to use tl.iterate.minibatches()
tl.utils.fit(sess, network, train_op, cost, X, Y, x, digits,
            acc=Acc, batch_size=44, n_epoch=500, print_freq=1,
            X_val=TestX, y_val=TestY, eval_train=False)

# Evaluation
tl.utils.test(sess, network, Acc, TestX, TestY, x, digits, batch_size=None, cost=cost)

# Save the network to .npz file
tl.files.save_npz(network.all_params , name='model.npz')

sess.close()
