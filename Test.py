# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow as tl

data = tl.files.load_npy_to_any(name = 'data.npy')
TestX= data['TestX']
TestY= data['TestY']
N_frames= data['N_frames']
nummfcc= data['nummfcc']
sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32,shape= [None,N_frames,nummfcc,1])
digits = tf.placeholder(tf.int64,shape=[None,])
 
network = tl.layers.InputLayer(x,name = 'MfccGraphInput')
network = tl.layers.Conv2d(network, n_filter=48, filter_size=(5, 5), strides=(1, 1),
            act=tf.nn.relu, padding='SAME', name='cnn1')
network = tl.layers.MaxPool2d(network, filter_size=(3, 3), strides=(3, 3),
            padding='SAME', name='pool1')
network = tl.layers.BatchNormLayer(network,name = 'BNLayer1')
network = tl.layers.Conv2d(network, n_filter=96, filter_size=(5, 5), strides=(1, 1),
            act=tf.nn.relu, padding='SAME', name='cnn2')
network = tl.layers.MaxPool2d(network, filter_size=(3, 3), strides=(3, 3),
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
sess.run(tf.initialize_all_variables())
load_params = tl.files.load_npz(name='model_test.npz')
tl.files.assign_params(sess, load_params, network)
# Evaluation
tl.utils.test(sess, network, Acc, TestX, TestY, x, digits, batch_size=None, cost=cost)