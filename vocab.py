# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 20:18:30 2014

@author: nicolai

An object to act as the "linear projection" layer in an unsupervised neural network

For use in semi-supervised representation learning
"""
import numpy as np
from random import choice


# Construct a look-up table containing the vector representations of the vocab
# It's a matrix of dimension p x |V|, where p is an arbitrary choice of 
# dimensionality (should be subject to cross validation)  

class vocabulary:
    """
    Init this class with a list of triplet tokens
    Each word will have a numerical ID, and the object can be queried with a 
    list of triplets, which will return a concatenated vector of triplet representations
    """
    
    def __init__(self, vocab, p):
        
        self.p = p
        # Matrix of triplet vectors
        # Each column is a representation of a triplet
        self.W = np.random.random((p,len(vocab)))
        
        # Set up vocabulary look-up table
        
        self.lookup = {}
        
        for i in range(len(vocab)):
            self.lookup[ vocab[i] ] = i
            
        # List of IDs of previously retrieved triplet vectors
            
        self.IDs = []
        
    def retrieve(self, list_of_triplets):
        """
        Give a list of triplets, get a concatenated vector of representations
        """
        
        self.IDs = []
        IDs = []
        
        for trip in list_of_triplets:
            
            IDs.append( self.lookup[trip.upper()] )
        
        self.IDs = IDs
        
        # Randomly sample the middle triplet to make x_hat
        
        ID_hat = list(IDs)
        
        proposals = np.asarray(self.lookup.values())
        indx = np.where(proposals == ID_hat[len(ID_hat)/2])
        proposals = np.delete(proposals, indx)
        ID_hat[len(ID_hat)/2] = choice(proposals)
        
        # Concatenate an output vector
        
        x = []
        
        for indx in IDs:
            
            x += self.W[:,indx].tolist()
        
        x_hat = []
        for indx in ID_hat:
            x_hat += self.W[:,indx].tolist()
        
        # Prepend bias
        
        x = [1] + x
        
        x_hat = [1] + x_hat
        
        return (np.asarray(x), np.asarray(x_hat))
        
        
    def update(self, updated_vector):
        """
        After adjusting the vector representations with backpropagation
        update the representation matrix W
        """
        
        # Remove bias
        
        new_vec = updated_vector[1:]
        
        # Each triplet is a substring of length p in this vector
        
        indx = 0        
        
        for ID in self.IDs:
            # Update the W column corresponding to this ID
            
            self.W[:,ID] = np.asarray( new_vec[indx:indx+self.p] )
            
            indx += self.p
            
            
