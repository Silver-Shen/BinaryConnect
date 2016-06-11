####################################
# 2016 Zheyan Shen                                   
# Appling Model Compression Method BinaryConnect                                  
# Onto Super Resolution Applicaion
#
####################################
import sys
import os
import time

import numpy as np
np.random.seed(1234)  # for reproducibility

import theano
import theano.tensor as T
import theano.sandbox.cuda
theano.sandbox.cuda.use('gpu1') 

import lasagne

import cPickle as pickle

import batch_norm
import binary_connect
from collections import OrderedDict
from scipy.io import loadmat,savemat

def inspect_inputs(i, node, fn):
    print i, node, "input(s) value(s):", [input[0] for input in fn.inputs]

def inspect_outputs(i, node, fn):
    print i, node, " output(s) value(s):", [output[0] for output in fn.outputs]

if __name__ == "__main__":
    #Parameters Setttings
    batch_size = 128
    print "batch_size = "+str(batch_size)
    
    # CNN parameters
    conv_num = 3
    conv_kernel = [9, 1, 5]
    conv_filters = [64, 32, 1]
    for i in range(conv_num):
        print "The %dth Convolutional layer has %dx%d kernel and %d feature maps" % (i+1, conv_kernel[i], conv_kernel[i], conv_filters[i])

    # Training parameters
    num_epochs = 300
    print "num_epochs = " + str(num_epochs)
    
    # BinaryConnect
    binary = True
    print "binary = " + str(binary)
    stochastic = True
    print "stochastic = " + str(stochastic)
    # (-H,+H) are the two binary values    
    H = 1.
    print "H = " + str(H)
    W_LR_scale = 1.    
    #W_LR_scale = "Glorot" # "Glorot" means we are using the coefficients from Glorot's paper
    print "W_LR_scale = " + str(W_LR_scale)
    
    # Decaying LR 
    LR_start = .0001
    print "LR_start = " + str(LR_start)
    LR_decay = 1
    print "LR_decay = " + str(LR_decay)
    
    print 'Loading SRCNN dataset...'
    # Load the training and validation data
    trainData = loadmat('train.mat')
    testData = loadmat('test.mat')
    x_train = trainData['x_train']    
    y_train = trainData['y_train']
    x_test = testData['x_test']
    y_test = testData['y_test']
    # Casting into [0, 1)
    x_train = x_train/255.
    y_train = y_train/255.
    x_test = x_test/255.
    y_test = y_test/255.
    print 'Building the CNN...'    

    # Prepare Theano variables for inputs and targets
    input = T.tensor4('inputs')
    target = T.tensor4('targets')
    LR = T.scalar('LR', dtype = theano.config.floatX)

    cnn = lasagne.layers.InputLayer(
            shape = (None, 1, 33, 33),
            input_var = input)
    
    for i in range(conv_num):
        if (i < conv_num - 1):            
            nonlinearity = lasagne.nonlinearities.rectify
        else:
            nonlinearity = lasagne.nonlinearities.identity
        cnn = binary_connect.Conv2DLayer(
                cnn, 
                binary = binary,
                stochastic = stochastic,
                H = H,
                W_LR_scale = W_LR_scale,
                num_filters = conv_filters[i],
                filter_size = (conv_kernel[i], conv_kernel[i]),
                pad = 0,                
                nonlinearity = nonlinearity)

    # Euclidean Loss
    train_output = lasagne.layers.get_output(cnn, deterministic = False)
    loss = lasagne.objectives.squared_error(train_output, target)
    loss = loss.mean()/2.

    if binary:        
        # W updates
        W = lasagne.layers.get_all_params(cnn, binary = True)
        W_grads = binary_connect.compute_grads(loss, cnn)
        updates = lasagne.updates.momentum(loss_or_grads = W_grads, params = W, learning_rate = LR, momentum = 0)
        updates = binary_connect.clipping_scaling(updates, cnn)
        
        # other parameters updates
        params = lasagne.layers.get_all_params(cnn, trainable = True, binary = False)
        updates = OrderedDict(updates.items() + lasagne.updates.momentum(loss_or_grads = loss, params = params, 
                                                                        learning_rate = LR, momentum = 0).items())
        
    else:
        params = lasagne.layers.get_all_params(cnn, trainable = True)
        updates = lasagne.updates.momentum(loss_or_grads = loss, params = params, learning_rate = LR, momentum = 0)

    test_output = lasagne.layers.get_output(cnn, deterministic = True)
    test_loss = lasagne.objectives.squared_error(test_output, target)
    test_loss = test_loss.mean()/2.    
    
    # Compile a function performing a training step on a mini-batch (by giving the updates dictionary) 
    # and returning the corresponding training loss:
    train_fn = theano.function([input, target, LR], [train_output, loss], updates = updates, allow_input_downcast = True)
                        # mode=theano.compile.MonitorMode(
                        # pre_func=inspect_inputs,
                        # post_func=inspect_outputs))

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input, target], [test_output, test_loss], allow_input_downcast = True)
                        # mode=theano.compile.MonitorMode(
                        # pre_func=inspect_inputs,
                        # post_func=inspect_outputs))

    print('Training...')
    
    binary_connect.train(
            train_fn = train_fn, val_fn = val_fn,
            batch_size = batch_size,
            LR_start = LR_start, LR_decay = LR_decay,
            num_epochs = num_epochs,
            val_interval = 10,
            X_train = x_train, y_train = y_train,            
            X_val = x_test, y_val = y_test)

    # Save the parameter
    W = lasagne.layers.get_all_param_values(cnn)
    savemat('x3_binary.mat',
            {'weights_conv1':W[0], 'bias_conv1':W[1],
             'weights_conv2':W[2], 'bias_conv2':W[3],
             'weights_conv3':W[4], 'bias_conv3':W[5]})