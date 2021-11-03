#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Selects high fidelity sequences and save them into several fragmented fasta files

@author: Octavio Gonzalez-Lugo
"""
###############################################################################
# Loading packages 
###############################################################################

import numpy as np
import multiprocessing as mp

from Bio import SeqIO
from io import StringIO

###############################################################################
# Global definitions
###############################################################################
#path of the NCBI SARS Cov 2 fasta sequences 
GlobalDirectory=r"/media/tavoglc/Datasets/DABE/LifeSciences/COVIDSeqs/sep2021/secuencias/"
seqDataDir=GlobalDirectory+'sequences.fasta'
seqMetaDataDir=GlobalDirectory+'sequences.csv'

sequencesFrags = GlobalDirectory + 'splitted/'

MaxCPUCount=int(0.6*mp.cpu_count())

###############################################################################
# Sequence Filtering Functions
###############################################################################

#Wrapper function to ask if the number of unique elements in a sequence is equal to 4 
def CanonicalAlphabetQ(sequence):
    if len(np.unique(sequence.seq))==4:
        return True
    else:
        return False
    
#Wrapper function to ask if the sequence is larger than 25000 elements 
def SizeQ(sequence):
    if len(sequence.seq)>25000:
        return True
    else:
        return False
    
def GetFilteredSeqsIndex(Sequences):
    '''
    Iterates over a series of DNA sequences to chek if the only avaliable bases 
    are A,C,G,T. 

    Parameters
    ----------
    Sequences : list
        list of strings, Contains the DNA sequences

    Returns
    -------
    canonical : list
        Index of the sequences that contains only four bases .
    noncanonical : list
        Index of the sequences that contains more than four bases. A different 
        character on a DNA sequence means that it could be one  of 2 or more bases 
        at that position

    '''
            
    localPool=mp.Pool(MaxCPUCount)
    canonicalResponce=localPool.map(CanonicalAlphabetQ,[val for val in Sequences])
    localPool.close()
    
    localPool=mp.Pool(MaxCPUCount)
    sizeResponce=localPool.map(SizeQ,[val for val in Sequences])
    localPool.close()
    
    canonical=[]
    
    for k,disc in enumerate(zip(canonicalResponce,sizeResponce)):
        
        if disc[0] and disc[1]:
            canonical.append(k)
        
    return canonical

###############################################################################
# Sequence Loading functions
###############################################################################

def batch_iterator(iterator, batch_size):
    """Returns lists of length batch_size.

    This can be used on any iterator, for example to batch up
    SeqRecord objects from Bio.SeqIO.parse(...), or to batch
    Alignment objects from Bio.AlignIO.parse(...), or simply
    lines from a file handle.

    This is a generator function, and it returns lists of the
    entries from the supplied iterator.  Each list will have
    batch_size entries, although the final list may be shorter.
    """
    entry = True  # Make sure we loop once
    while entry:
        batch = []
        while len(batch) < batch_size:
            try:
                entry = next(iterator)
            except StopIteration:
                entry = None
            if entry is None:
                # End of file
                break
            batch.append(entry)
        if batch:
            yield batch

record_iter = SeqIO.parse(open(seqDataDir), "fasta")
for i, batch in enumerate(batch_iterator(record_iter, 50000)):
    filename = sequencesFrags + "group_%i.fasta" % (i + 1)
    sequenceIndex = GetFilteredSeqsIndex(batch)
    newbatch = [batch[k] for k in sequenceIndex]
    
    with open(filename, "w") as handle:
        count = SeqIO.write(newbatch, handle, "fasta")
    print("Wrote %i records to %s" % (count, filename))

