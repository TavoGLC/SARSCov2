#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MIT License
Copyright (c) 2021 Octavio Gonzalez-Lugo 

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
@author: Octavio Gonzalez-Lugo
"""

###############################################################################
# Loading packages 
###############################################################################

import os 
import numpy as np
import pandas as pd
import networkx as nx
import multiprocessing as mp
import matplotlib.pyplot as plt


from Bio import SeqIO
from io import StringIO

from itertools import product

from sklearn.decomposition import PCA
from sklearn import preprocessing as pr

###############################################################################
# Utility Functions
###############################################################################

def GetGridShape(TotalNumberOfElements):
    """
    Parameters
    ----------
     TotalNumberOfElements : int
        Total number of elements in the plot.

    Returns
    -------
    nrows : int
        number of rows in the plot.
    ncolumns : int
        number of columns in the plot.

    """
    numberOfUnique=TotalNumberOfElements
    squaredUnique=int(np.sqrt(numberOfUnique))
    
    if squaredUnique*squaredUnique==numberOfUnique:
        nrows,ncolumns=squaredUnique,squaredUnique
    elif squaredUnique*(squaredUnique+1)<numberOfUnique:
        nrows,ncolumns=squaredUnique+1,squaredUnique+1
    else:
        nrows,ncolumns=squaredUnique,squaredUnique+1
    
    return nrows,ncolumns

#Wrapper function, flatten a list of lists
def ListFlatten(List):
    return [item for sublist in List for item in sublist]

###############################################################################
# Visualization functions
###############################################################################

def PlotStyle(Axes): 
    """
    Applies a general style to a plot 
    Parameters
    ----------
    Axes : Matplotlib axes object
        Applies a general style to the matplotlib object

    Returns
    -------
    None.
    """    
    Axes.spines['top'].set_visible(False)
    Axes.spines['bottom'].set_visible(True)
    Axes.spines['left'].set_visible(True)
    Axes.spines['right'].set_visible(False)
    Axes.xaxis.set_tick_params(labelsize=13)
    Axes.yaxis.set_tick_params(labelsize=13)
    
def MakePCAPlot(Data,ax):
    '''
    Creates a plot of the first and second principal components in the data

    Parameters
    ----------
    Data : array-like
        Data for PCA projection.
    ax : matplotlib axes object
        Axes to place the plot.

    Returns
    -------
    None.

    '''
    
    MethodPCA=PCA(n_components=2)
    MethodPCA.fit(Data)
    DataPCA=MethodPCA.transform(Data)
    
    ax.plot(DataPCA[:,0],DataPCA[:,1],'bo',alpha=0.05)
    ax.set_xlabel('First Principal Component',fontsize=12)
    ax.set_ylabel('Second Principal Component',fontsize=12)
        

#Wrapper function to create a panel of MakePCAPlot
def MakePCAPanel(Data,Titles):
    
    nrows,ncolumns=GetGridShape(len(Data))
    
    fig,axes=plt.subplots(nrows,ncolumns,figsize=(15,15))
    
    if nrows==1 or ncolumns==1:
        maxInx=max([nrows,ncolumns])
        subPlotIndexs=[j for j in range(maxInx)]
    else:
        subPlotIndexs=[(j,k) for j in range(nrows) for k in range(ncolumns)]
        
    for k ,dset in enumerate(Data):
        cInx=subPlotIndexs[k]
        MakePCAPlot(dset,axes[cInx])
        axes[cInx].set_title(Titles[k],fontsize=12)

    for val in subPlotIndexs:
        PlotStyle(axes[val])

###############################################################################
# Sequence Loading functions
###############################################################################

#Wrapper function to load the sequences
def GetSeqs(Dir):
    
    cDir=Dir
    
    with open(cDir) as file:
        
        seqData=file.read()
        
    Seq=StringIO(seqData)
    SeqList=list(SeqIO.parse(Seq,'fasta'))
    
    return SeqList

###############################################################################
# Sequence K-mer generating functions
###############################################################################

def SplitString(String,ChunkSize):
    '''
    Split a string ChunkSize fragments using a sliding windiow

    Parameters
    ----------
    String : string
        String to be splitted.
    ChunkSize : int
        Size of the fragment taken from the string .

    Returns
    -------
    Splitted : list
        Fragments of the string.

    '''
    try:
        localString=str(String.seq)
    except AttributeError:
        localString=str(String)
      
    if ChunkSize==1:
        Splitted=[val for val in localString]
    
    else:
        nCharacters=len(String)
        Splitted=[localString[k:k+ChunkSize] for k in range(nCharacters-ChunkSize)]
        
    return Splitted

def UniqueToDictionary(UniqueElements):
    '''
    Creates a dictionary that takes a Unique element as key and return its 
    position in the UniqueElements array
    Parameters
    ----------
    UniqueElements : List,array
        list of unique elements.

    Returns
    -------
    localDictionary : dictionary
        Maps element to location.

    '''
    
    localDictionary={}
    nElements=len(UniqueElements)
    
    for k in range(nElements):
        localDictionary[UniqueElements[k]]=k
        
    return localDictionary

###############################################################################
# Sequences as graphs. 
###############################################################################

def MakeSequenceGraph(Sequence,NodeNames,viz=False):
    '''

    Parameters
    ----------
    Sequence : str, sequence object
        Sequence used to construct the graph.
    NodeNames : list, array-like
        Contains the name of the nodes of the graph.
    viz : bool, optional
        If true a Graph networkx object is created for fast visualization. 
        The default is False.

    Returns
    -------
    localGraph : networkx graph object
        De Brujins graph representation of the sequence.

    '''
    
    fragmentSize=len(NodeNames[0])
    processedSequence=SplitString(Sequence,fragmentSize)
    Nodes=np.arange(len(NodeNames))
    localDict=UniqueToDictionary(NodeNames)
    
    if viz:
        localGraph=nx.Graph()
    else:    
        localGraph=nx.MultiGraph()
        
    localGraph.add_nodes_from(Nodes)
    
    for k in range(len(processedSequence)-1):
        
        if processedSequence[k] in NodeNames and processedSequence[k+1] in NodeNames:
        
            current=localDict[processedSequence[k]]
            forward=localDict[processedSequence[k+1]]
            localGraph.add_edge(current,forward)
        
    return localGraph

###############################################################################
# Sequences as graphs. 
###############################################################################

#Wrapper function to calculate the node degree
def GetNormalizedDegreeData(Sequence,Nodes):
    localGraph=MakeSequenceGraph(Sequence,Nodes)
    localDegree=np.array(nx.degree(localGraph))
    return list(localDegree[:,1]/np.sum(localDegree[:,1]))

#Wraper function for parallelization 
def GetDataParallel(DataBase,Nodes,Function):
    
    localPool=mp.Pool(MaxCPUCount)
    graphData=localPool.starmap(Function, [(val.seq,Nodes )for val in DataBase])
    localPool.close()
    
    return graphData

###############################################################################
# Global definitions
###############################################################################

GlobalDirectory=r"/media/tavoglc/storage/storage/datasets/DABE/LifeSciences/COVIDSeqs/nov2021"
SequencesDir = GlobalDirectory + "/secuencias"  

seqDataDir=SequencesDir+'/sequences.fasta'
sequencesFrags = SequencesDir + '/splitted'

MaxCPUCount=int(0.85*mp.cpu_count())

###############################################################################
# Blocks
###############################################################################

Alphabet = ['A','C','T','G']
UniqueElementsBlock = []

maxSize = 5
for k in range(1,maxSize):
    
    UniqueElementsBlock.append([''.join(i) for i in product(Alphabet, repeat = k)])
    
Lenghts=[len(val) for val in UniqueElementsBlock]
Lenghts=np.cumsum(Lenghts)

fragmentsDirs = [sequencesFrags + '/' + dr for dr in os.listdir(sequencesFrags)]

###############################################################################
# Blocks
###############################################################################

names = []
for k ,block in enumerate(UniqueElementsBlock):
    j=0
    for val in fragmentsDirs:
    
        currentSeqs = GetSeqs(val)
        names = names + [val.id for val in currentSeqs]
        
        if len(currentSeqs)!=0:
            currentProcess = np.array(GetDataParallel(currentSeqs,block,GetNormalizedDegreeData))
            
            if j==0:
                localData=currentProcess
            else:
                localData=np.vstack((localData,currentProcess))
            j = j+1
        else:
            j = j+1
    
    if k==0:
        KmerData=localData
    else:
        KmerData=np.hstack((KmerData,localData))


Kmers = KmerData

Scaler=pr.MinMaxScaler()
Scaler.fit(Kmers)
Kmers=Scaler.transform(Kmers)
MakePCAPanel([Kmers[:,0:Lenghts[0]],Kmers[:,0:Lenghts[1]],Kmers[:,0:Lenghts[2]],Kmers[:,0:Lenghts[3]]],['1-mer','1-mer+2-mer','1-mer+2-mer+3-mer','1-mer+2-mer+3-mer+4-mer'])

###############################################################################
# Blocks
###############################################################################
  
sequenceNames = []

for val in fragmentsDirs:
    
    currentSeqs = GetSeqs(val)
    sequenceNames = sequenceNames + [val.id for val in currentSeqs]
    
KmerDF = pd.DataFrame()
KmerDF['id'] = np.array(sequenceNames)
headers = [val for li in UniqueElementsBlock for val in li]

for k,hd in enumerate(headers):
    KmerDF[hd] = Kmers[:,k]
    
KmerDF.to_csv(GlobalDirectory+'/KmerData.csv',index=False)