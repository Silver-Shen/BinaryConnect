# Copyright 2015 Matthieu Courbariaux

# This file is part of BinaryConnect.

# BinaryConnect is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# BinaryConnect is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with BinaryConnect.  If not, see <http://www.gnu.org/licenses/>.

import time
from collections import OrderedDict
import numpy as np

# specifying the gpu to use
import theano.sandbox.cuda
theano.sandbox.cuda.use('gpu1') 
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import lasagne

# Thresholding between 0 and 1
def hard_sigmoid(x):
    return T.clip((x+1.)/2., 0, 1)

# The binarization function
def binarization(W, H, binary = True, deterministic = False, stochastic = False, srng = None):    
    # (deterministic == True) <-> test-time <-> inference-time
    if not binary or (deterministic and stochastic):        
        Wb = W    
    else:        
        # [-1,1] -> [0,1]
        Wb = hard_sigmoid(W/H)        
        # Stochastic BinaryConnect
        if stochastic:                    
            Wb = T.cast(srng.binomial(n=1, p=Wb, size=T.shape(Wb)), theano.config.floatX)
        # Deterministic BinaryConnect (round to nearest)
        else:            
            Wb = T.round(Wb)       
        # 0 or 1 -> -1 or 1
        Wb = T.cast(T.switch(Wb, H, -H), theano.config.floatX)    
    return Wb

# This class extends the Lasagne DenseLayer to support BinaryConnect
class DenseLayer(lasagne.layers.DenseLayer):
    
    def __init__(self, incoming, num_units, 
        binary = True, stochastic = True, H = 1., W_LR_scale = "Glorot", **kwargs):        
        self.binary = binary
        self.stochastic = stochastic
        # Two binarized value(default to be +1 & -1)
        self.H = H
        if H == "Glorot":
            num_inputs = int(np.prod(incoming.output_shape[1:])) # first dimension be the batchsize?
            self.H = np.float32(np.sqrt(1.5/(num_inputs+num_units))) # more fan-in, more narrow interval           
            
        self.W_LR_scale = W_LR_scale
        if W_LR_scale == "Glorot":
            num_inputs = int(np.prod(incoming.output_shape[1:]))
            self.W_LR_scale = np.float32(1./np.sqrt(1.5/(num_inputs+num_units)))

        self._srng = RandomStreams(lasagne.random.get_rng().randint(1, 2147462579))
        
        if self.binary:
            super(DenseLayer, self).__init__(incoming, num_units, W=lasagne.init.Uniform((-self.H,self.H)), **kwargs)
            # add the binary tag to weights            
            self.params[self.W]=set(['binary'])            
        else:
            super(DenseLayer, self).__init__(incoming, num_units, **kwargs)
        
    def get_output_for(self, input, deterministic=False, **kwargs):
        
        self.Wb = binarization(self.W, self.H, self.binary, deterministic, self.stochastic, self._srng)
        Wr = self.W # backup the precise value of weight
        self.W = self.Wb
            
        rvalue = super(DenseLayer, self).get_output_for(input, **kwargs)
        
        self.W = Wr # restore weight        
        return rvalue

# This class extends the Lasagne Conv2DLayer to support BinaryConnect
class Conv2DLayer(lasagne.layers.Conv2DLayer):
    
    def __init__(self, incoming, num_filters, filter_size,
        binary = True, stochastic = True, H = 1., W_LR_scale = "Glorot", **kwargs):        
        self.binary = binary
        self.stochastic = stochastic
        
        self.H = H
        if H == "Glorot":
            num_inputs = int(np.prod(filter_size)*incoming.output_shape[1])
            num_units = int(np.prod(filter_size)*num_filters) # theoretically, I should divide num_units by the pool_shape
            self.H = np.float32(np.sqrt(1.5 / (num_inputs + num_units)))            
        
        self.W_LR_scale = W_LR_scale
        if W_LR_scale == "Glorot":
            num_inputs = int(np.prod(filter_size)*incoming.output_shape[1])
            num_units = int(np.prod(filter_size)*num_filters) # theoretically, I should divide num_units by the pool_shape
            self.W_LR_scale = np.float32(1./np.sqrt(1.5 / (num_inputs + num_units)))            
            
        self._srng = RandomStreams(lasagne.random.get_rng().randint(1, 2147462579))
            
        if self.binary:            
            super(Conv2DLayer, self).__init__(incoming, num_filters, filter_size, W=lasagne.init.Uniform((-self.H,self.H)), **kwargs)   
            # add the binary tag to weights            
            self.params[self.W]=set(['binary'])
        else:
            super(Conv2DLayer, self).__init__(incoming, num_filters, filter_size, **kwargs)    
    
    # def convolve(self, input, deterministic=False, **kwargs):        
    #     self.Wb = binarization(self.W, self.H, self.binary, deterministic, self.stochastic, self._srng)
    #     Wr = self.W
    #     self.W = self.Wb
            
    #     rvalue = super(Conv2DLayer, self).convolve(input, **kwargs)
        
    #     self.W = Wr        
    #     return rvalue

    # Experimental Function
    def get_output_for(self, input, deterministic=False, **kwargs):       
        #self.W = theano.printing.Print('Float weight: ')(self.W)         
        self.Wb = binarization(self.W, self.H, self.binary, deterministic, self.stochastic, self._srng)        
        Wr = self.W # backup the precise value of weight
        self.W = self.Wb
        #self.W = theano.printing.Print('Binary weight: ')(self.W)
        rvalue = super(Conv2DLayer, self).get_output_for(input, **kwargs)        
        self.W = Wr # restore weight        
        return rvalue

# This function computes the gradient of the binary weights
def compute_grads(loss, network):
    layers = lasagne.layers.get_all_layers(network)
    grads = []
    
    for layer in layers:
        params = layer.get_params(binary = True)                
        if params:            
            grads.append(theano.grad(loss, wrt=layer.Wb))

    return grads

# This functions clips the weights after the parameter update
def clipping_scaling(updates, network):
    
    layers = lasagne.layers.get_all_layers(network)
    updates = OrderedDict(updates)
    
    for layer in layers:
    
        params = layer.get_params(binary=True)
        for param in params:
            print("W_LR_scale = "+str(layer.W_LR_scale))
            print("H = "+str(layer.H))
            updates[param] = param + layer.W_LR_scale*(updates[param] - param)
            updates[param] = T.clip(updates[param], -layer.H, layer.H)     

    return updates
        
# Given a dataset and a model, this function trains the model on the dataset for several epochs
# (There is no default train function in Lasagne yet)
def train(train_fn,val_fn,
            batch_size,
            LR_start,LR_decay,
            num_epochs,
            val_interval,
            X_train,y_train,
            X_val,y_val,
            X_test=None,y_test=None):
    
    # A function which shuffles a dataset
    def shuffle(X, y):    
        shuffled_range = range(len(X))
        np.random.shuffle(shuffled_range)                
        new_X = np.copy(X)
        new_y = np.copy(y)        
        for i in range(len(X)):            
            new_X[i] = X[shuffled_range[i]]
            new_y[i] = y[shuffled_range[i]]

        return new_X,new_y
    
    # This function trains the model a full epoch (on the whole dataset)
    def train_epoch(X, y, LR):        
        loss = 0
        batches = len(X)/batch_size                
        for i in range(batches):            
            output, new_loss = train_fn(X[i*batch_size:(i+1)*batch_size], y[i*batch_size:(i+1)*batch_size], LR)            
            loss += new_loss

        loss /= batches        
        return loss
    
    # This function tests the model a full epoch (on the whole dataset)
    def val_epoch(X, y):        
        #err = 0
        loss = 0
        batches = len(X)/batch_size        
        for i in range(batches):
            #new_loss, new_err = val_fn(X[i*batch_size:(i+1)*batch_size], y[i*batch_size:(i+1)*batch_size])
            new_output, new_loss = val_fn(X[i*batch_size:(i+1)*batch_size], y[i*batch_size:(i+1)*batch_size])           
            #err += new_err
            loss += new_loss
        
        #err = err / batches * 100
        loss /= batches
        #return err, loss
        return loss
    
    # shuffle the train set
    X_train, y_train = shuffle(X_train,y_train)
    #best_val_err = 100
    best_epoch = 1
    LR = LR_start
    
    # We iterate over epochs:
    for epoch in range(num_epochs):

        start_time = time.time()        
        train_loss = train_epoch(X_train, y_train, LR)
        X_train,y_train = shuffle(X_train, y_train)
        
        #val_err, val_loss = val_epoch(X_val,y_val)
        if (epoch % val_interval == 0):
            val_loss = val_epoch(X_val, y_val)
        
        # test if validation error went down
        # if val_err <= best_val_err:            
        #     best_val_err = val_err
        #     best_epoch = epoch + 1          
        #     if X_test != None:
        #         test_err, test_loss = val_epoch(X_test, y_test)                
        
        epoch_duration = time.time() - start_time
        
        # Then we print the results for this epoch:
        print("Epoch "+str(epoch + 1)+" of "+str(num_epochs)+" took "+str(epoch_duration)+"s")
        print("  LR:                            "+str(LR))
        print("  training loss:                 "+str(train_loss))
        if (epoch % val_interval == 0):
            print("  validation loss:               "+str(val_loss))
        #print("  validation error rate:         "+str(val_err)+"%")
        #print("  best epoch:                    "+str(best_epoch))
        #print("  best validation error rate:    "+str(best_val_err)+"%")
        if X_test != None and y_test != None:
            print("  test loss:                     "+str(test_loss))
            print("  test error rate:               "+str(test_err)+"%") 
        
        # decay the LR
        LR *= LR_decay