# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 12:51:26 2014

@author: nicolai

Scripts to compile data sets from pre-specified context window sizes
"""
import numpy as np

def read_FASTA(fname):
    '''Reads in a FASTA file and returns the sequence.'''
    lines = open(fname).readlines()[1:]
    return ''.join( line.strip() for line in lines)
    
    
def windows(seqfile, annfile, winsize = 20):    
    '''Iterates through a FASTA sequence together with it's gene
    annotation, providing each window together with the annotation
    window.'''
    
    seq = read_FASTA(seqfile)
    ann = read_FASTA(annfile)
    
    assert len(seq) == len(ann)
    
    for i in xrange( len(seq) / winsize ):
        yield seq[i*winsize:(i+1)*winsize], ann[i*winsize:(i+1)*winsize]

    # The last part of the sequence isn't a full window, but we still need it
    #yield seq[(i+1)*winsize:], ann[(i+1)*winsize:]
    
def getKmers(seq, k):
    """
    Extract all k-mers in a dictionary
    """
    
    kmd = {}
    
    for i in range(len(seq)+1-k):
        kmer = seq[i:i+k]
        kmd[kmer] = kmd.get(kmer,0) + 1
    return kmd
    
def get_triplet_composition(seq):
    """
    Given a sequence (let's say from a context window), extract
    its components under the assumption that each "word"
    in the sequence is a triplet, and triplets may overlap on the
    last base
    """
    out = []    
    for i in range(len(seq)):
        
        if i+3 > len(seq):
            break
        out.append(seq[i:i+3])
    return out

class dataset:
    """
    Init this class with the AnnFiles,SeqFiles pair
        
    Will ensure that one context window always results in the same amount
    of triplets (this is ad hoc :( )
    """
    
    def __init__(self, seqfiles, annfiles):
        self.data = []
        # Run through dataset windows here
        
        for i in range(len(seqfiles)):
            for seqwin,annwin in windows(seqfiles[i],annfiles[i]):
                x = get_triplet_composition(seqwin)
                
                content = set(annwin)
                # One-of-K encoding for annotation windows
                if len(content) > 1:
                    # Window contains coding and non-coding regions
                    # This is a transition
                    y = np.asarray([0,0,1])
                else:
                    if "N" in content:
                        # Non-coding window"
                        y = np.asarray([1,0,0])
                    elif "R" in content or "C" in content:
                        # Coding window
                        y = np.asarray([0,1,0])
                self.data.append( (x,y) )
        self.S = read_FASTA(seqfiles[0])
