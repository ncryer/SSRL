# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 16:30:46 2014

@author: nicolai

Handle data operations here
"""
import glob
from make_dataset import *
from model import *
from vocab import *

class trainer:
    """
    Handle the various training methods here
    """
    def __init__(self, dataset, split = [70,30], vector_repr_dim = 30):
        
        assert sum(split) == 100
        
        self.p = vector_repr_dim
        
        self.dataset = dataset
        
        self.L = len(dataset.data)
        
        self.trainIndx = int(np.round(self.L/70))
        
        self.trainset = dataset.data[:self.trainIndx]
        
        self.testset = dataset.data[self.trainIndx:]
        
        # Setup trainer objects: vocabulary and network
        
        self.vocab = self.setup_vocab()
        
        self.NN = self.setup_network()
        
        # Statistics
        
        self.error_unsupervised = []
        self.error_supervised = []
        self.empirical_risk = []
        
    
    def setup_vocab(self):
        """
        Construct a look-up table for vector representations
        
        Currently the vocabulary is hard-coded to only consider triplets,
        but in principle any k-mer can be used - just keep in mind the
        resulting data sparsity of big k-mers. 
        
        The vocabulary is fed to the neural network object and updated intern-
        ally.
        """
        
        # Get all possible triplet tokens
        print "Setting up vocabulary...\n"
        
        voc = getKmers(self.dataset.S, 3).keys()
        
        vocab = vocabulary(voc, self.p)
        
        print "Done.\n"
        
        return vocab
        
        
    def setup_network(self, number_of_hidden_units = False, size_of_output = 3, activation = "relu"):
        """
        Construct a neural network object with the following functions:
        
        nn.SGD_unsupervised(input):
        input is a tuple (x, x_hat), comes directly from self.vocab.retrieve()
        
        Silently updates the weights and the vector representations
        
        --------------------------------------------------------------
        
        nn.SGD_supervised(input):
        input is a tuple (x,y) where y is the one-of-K encoding for the dataset
        this is simply each element in the training and testing sets
        
        Silently updates the weights and vector representations
        
        --------------------------------------------------------------
        
        nn.predict(input):
        Perform a winner-takes-all classification by doing one feed-forward
        pass with a softmax output layer
        
        Returns 1 if correctly classified, 0 else
        
        --------------------------------------------------------------
        
        Recommended number of hidden units is 10% of input size, but should
        be subject to cross-validation. 
        
        Choice of non-linearity defaults to rectified linear units, since these
        seem to be best at lowering the error
        """
        
        inp_size = len(self.vocab.retrieve(self.trainset[0][0])[0])
        
        if not number_of_hidden_units:
            n_hidden = int(np.round(inp_size * 0.1))
        
        return FFNet(inp_size, n_hidden, size_of_output, self.vocab, activation = "relu")
        
    def unsupervised_pretraining(self, number_of_epochs=1):
        """
        Train the vector representations to (hopefully) be context-aware
        
        This is strongly recommended prior to any actual classification task
        
        One epoch corresponds to one complete run through the training set
        """
        
        for epoch in range(number_of_epochs):
            
            for i in range(len(self.trainset)):
                inpTuple = self.NN.triplet_representations.retrieve(self.trainset[i][0])
                self.NN.SGD_unsupervised(inpTuple)
            
            err = np.mean(self.NN.errors)
            self.error_unsupervised.append(err)
            print "Mean error: " + str(err)
            self.NN.errors = []
            
    def supervised_pretraining(self, number_of_epochs=1):
        """
        Backpropagate label information into the vector representations
        """
        
        for epoch in range(number_of_epochs):
            
            for i in range(len(self.trainset)):
                x,y = self.trainset[i]
                # Need only the x representation, not x_hat
                x = self.NN.triplet_representations.retrieve(x)[0]
                
                self.NN.SGD_supervised((x,y))
                
            err = np.mean(self.NN.supervised_errors)
            print "Mean error: " + str(err)
            self.error_supervised.append(err)
            
            self.NN.supervised_errors = []
    
    def predict_all(self):
        """
        Once trained, evaluate classification error on training and test sets
        """
        trainres = []        
        testres = []
        for i in range(len(self.trainset)):
            x,y = self.trainset[i]
            # Need only the x representation, not x_hat
            x = self.NN.triplet_representations.retrieve(x)[0]
            
            trainres.append(self.NN.predict((x,y)))
        
        for i in range(len(self.testset)):
            x,y = self.testset[i]
            # Need only the x representation, not x_hat
            x = self.NN.triplet_representations.retrieve(x)[0]
            
            testres.append(self.NN.predict((x,y)))
        
        trainerr = sum(trainres)/float(len(trainres))
        testerr = sum(testres)/float(len(testres))
        
        print "Training error: " + str(trainerr) + "\n"
        print "Test error: " + str(testerr) + "\n"
        
        
        
if __name__ == "__main__":
    AnnFiles = glob.glob("annotation*.fa")
    AnnFiles.sort()

    SeqFiles = glob.glob("genome*.fa")
    SeqFiles.sort()
    print "Building dataset, takes a little while..."
    myData = dataset(SeqFiles, AnnFiles)
    print "Done."
    
    myTrainer = trainer(myData)
    
    # Without pre-training
    #myTrainer.predict_all()
    # With
    myTrainer.unsupervised_pretraining(5)

    myTrainer.predict_all()


    
