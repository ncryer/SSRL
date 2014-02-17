# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 16:37:51 2014

@author: nicolai
"""
import numpy as np 

@np.vectorize
def sigmoid(z):
        
    return 1./(1. + np.exp(-z))       

class FFNet:
    """
    A single-layer feed-forward neural network to be trained with hinge loss
    """
    def __init__(self, size_of_input, size_of_hidden_layer, size_of_output, vocabulary, activation = "sigmoid", learning_rate = 0.1, momentum = 0.2):
        
        # Input
        self.n_x = size_of_input
        # Output
        self.n_y = size_of_output
        # hidden layer
        self.n_hid = size_of_hidden_layer
        # input-to-hidden matrix of weights
        self.W = np.random.uniform(-0.1,0.1, (self.n_hid, self.n_x))
        # Hidden to softmax matrix 
        self.W_out = np.random.uniform(-0.1,0.1, (self.n_y, self.n_hid))
        # Random vector to compute score
        self.U = np.asarray( [1./self.n_hid]*self.n_hid )
        
        # Non-linearity of choice
        
        act_functions = { "sigmoid" : self.sigmoid, "tanh" : self.tanh, "relu" : self.ReLU }
        gradients = {"sigmoid" : self.siggrad, "tanh" : self.tanhgrad, "relu": self.ReLUgrad}
        
        self.act_fun = act_functions[ activation ]
        self.grad_fun = gradients[ activation ]
        
        # Learning rate, momentum
        self.learning_rate = learning_rate
        self.momentum = momentum
        
        # triplet representation look-up table
        self.triplet_representations = vocabulary
        
        # Statistics
        self.errors = []
        
        self.supervised_errors = []
        
        self.predict_errors = []
        
     
    # Non-linear activation functions
    @np.vectorize
    def sigmoid(z):
        
        return 1./(1. + np.exp(-z))
    
    @np.vectorize
    def siggrad(z):
        y = sigmoid(z)
        return y*(1-y)
    
    # Only ReLU seems to work well
    def tanh(self, z):
        
        u = np.exp(2*z)
        
        return (2*u - 1)/(2*u - 1)
        
    def tanhgrad(self,z):
        
        return 1 - (self.tanh(z) ** 2)
    
    def ReLU(self, z):
        # Rectified linear unit
        out = np.zeros(len(z))
        for i in range(len(z)):
            out[i] = max(0,z[i])
        return out
    
    def ReLUgrad(self,z):
        # Return gradient of rectified linear unit
        # This is wrong, needs to be applied element-wise
        out = np.zeros( len(z) )
        for i in range(len(z)):
            if max(0,z[i]) != 0:
                out[i] = 1
        return out
        
    def softmax(self, x, weightmat):
        # Softmax activation function for classification
        Z = np.dot(weightmat, x)
        numerator = np.exp(Z)
        out = numerator / np.sum(numerator)
        gradient = out*(1-out)
        return out, gradient
    
    def SGD_unsupervised(self, inpTuple):
        """
        Run forward propagation, compute and change weights and inputs
        if error > 0
        
        inpTuple contains (x,x_hat)
        """
        x = inpTuple[0]
        x_hat = inpTuple[1]        
        
        # Forward pass
        z = np.dot(self.W,x)
        s_x = np.dot( self.U, self.act_fun( z ) )
        s_x_hat = np.dot( self.U, self.act_fun( np.dot(self.W, x_hat) ) )
        
        # Use hinge loss error
        error = max(0, 1 - s_x + s_x_hat)
        
        if error > 0:
            # Backward pass to change weights and representations
            delta = self.grad_fun( z ) * self.U
            
            d_W = np.outer(delta, x)
            
            # Update weight matrix
            
            self.W = self.W + self.learning_rate * d_W
            
            # Update representation vector
            
            x = x + self.learning_rate * np.dot(self.W.T, delta)
            
        
            self.errors.append(error)
            
            # Update dictionary with new x
            
            self.triplet_representations.update( x )
            
            
    def SGD_supervised(self, inpTuple):
        """
        Stochastic gradient descent with a softmax output
 
        """
        x = inpTuple[0]
        y = inpTuple[1]
        
        # ---------- Forward pass ----------------
        
        # Activities in hidden layer
        z_hid = np.dot(self.W,x)
        a_hid = self.act_fun(z_hid)
        
        z_out = np.dot(self.W_out, a_hid)
        
        # Softmax output
        h = np.exp(z_out)
        h = h / np.sum(h)
  

        
        # --------- Backward pass -----------
        
        # Output delta - softmax gradient
        f_prime = h * (1-h)
        
        delta_2 = -(y - h) * f_prime
        
        delta_1 = np.dot(self.W_out.T, delta_2) * self.grad_fun( z_hid )
        
        # Gradient updates
        
        grad_W_out = np.zeros(self.W_out.shape)
        grad_W = np.zeros(self.W.shape)
        
        grad_W_out = grad_W_out + np.outer(delta_2, a_hid)
        grad_W = grad_W + np.outer(delta_1, x)
        
        self.W_out = self.W_out - self.learning_rate * grad_W_out
        self.W = self.W - self.learning_rate * grad_W
        
        # Update x
        x = x - self.learning_rate * np.dot(self.W.T, delta_1)
        
        self.triplet_representations.update( x )
        
        # Statistics
        # Cross-entropy error
        
        error = - sum(y * np.log(h))
        
        self.supervised_errors.append( error )
    
    def predict(self, inpTuple):
        """
        After training the network, use this function to make predictions
        
        Is equivalent to simply running a forward pass with a softmax output
        
        Output is a boolean: 1 if correct classification, 0 else
        """
        x = inpTuple[0]
        y = inpTuple[1]        
        
        z_hid = np.dot(self.W,x)
        a_hid = self.act_fun(z_hid)
        
        z_out = np.dot(self.W_out, a_hid)
        
        # Softmax output
        # Protect from under/overflow
        z = np.maximum(z_out, -1e3)
        z = np.minimum(z, 1e3)
        
        # TODO: THIS NEEDS TO BE FIXED
        try:
            h = np.exp(z)
        except RuntimeWarning:
            return 0
        h = h / np.sum(h)
  
        # Cross-entropy error
        
        error = - sum(y * np.log(h))

        # Update stats

        self.predict_errors.append( error )

        # Return a winner-takes-all prediction
        prediction = np.asarray([0,0,0])
        prediction[np.argmax(h)] = 1
        
        if False in (prediction == y):
            # Misclassified
            return 0
        else:
            return 1