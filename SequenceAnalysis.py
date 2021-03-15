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

import re
import numpy as np
import pandas as pd
import multiprocessing as mp
import matplotlib.pyplot as plt

from Bio import SeqIO
from io import StringIO

from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn import preprocessing as pr

from scipy.spatial import distance as ds 

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
def Flatten(List):
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
    
def MakeSimplePlotFromDF(DataFrame,kind='Line'):
    '''
    Makes a simple plot from a data frame, data in the data frame must be numerical 
    Parameters
    ----------
    DataFrame : pandas data frame 
        data for the plot.
    kind : string, optional
        Controlls the kind of plot to be used. The default is 'Line'.

    Returns
    -------
    None.

    '''
    
    labels=DataFrame.keys()
    xs=np.arange(len(labels))
    
    plt.figure(figsize=(15,5))
    
    if kind=='Line':    
        plt.plot(xs,DataFrame)
    elif kind=='Bar':
        plt.bar(xs,DataFrame)
        
    ax=plt.gca()    
    ax.set_xticks(xs)
    ax.set_xticklabels(labels,rotation=85,fontsize=14)
    PlotStyle(ax)
        
def MakeClustersPlot(Data,Labels,ax):
    '''
    Scater plot of previoulsy calculated clusters. Each cluster have an 
    evenly spaced different color. 
    Parameters
    ----------
    Data : array
        Data For the scatter plot.
    Labels : array
        Same length as Data, contains the label for each sample.
    ax : matplot lib axes object
        Axes for the plot.

    Returns
    -------
    None.

    '''
    UniqueClusters=np.unique(Labels)
    colors=[plt.cm.viridis(val,alpha=0.25) for val in np.linspace(0,1,num=UniqueClusters.size-1)]
    plt.figure()
    
    for val in UniqueClusters:
        
        Xdata=[Data[k,0] for k in range(len(Data)) if Labels[k]==val]
        Ydata=[Data[k,1] for k in range(len(Data)) if Labels[k]==val]
        
        if val==-1:    
            ax.plot(Xdata,Ydata,'ko')
        else:      
            ax.plot(Xdata,Ydata,'o',color=colors[val])
        
    ax.set_xlabel('First Principal Component',fontsize=12)
    ax.set_ylabel('Second Principal Component',fontsize=12)


def MakePlotPanel(Data,config):
    '''
    Makes a panel of three plots: 
    -Scater plot of the PCA projection of the data
    -Ordered distances for cluster min distance determination 
    -Plot of each cluster with different colors 
    
    Parameters
    ----------
    Data : array
        Data to be analyzed.
    config : list
        Configuration of the DBSCAN algorithm EPS and min_samples.

    Returns
    -------
    ClusterLabels : array
        Cluster labels.

    '''
    
    MethodPCA=PCA(n_components=2)
    MethodPCA.fit(Data)
    DataPCA=MethodPCA.transform(Data)
    
    Distances=ds.pdist(DataPCA,metric='euclidean')
    Squared=ds.squareform(Distances)
    Squared=[np.sort(val) for val in Squared]
    Squared=np.array(Squared)
    
    EPS,MIN=config
    ClusterData=DBSCAN(eps=EPS,min_samples=MIN).fit(DataPCA)
    ClusterLabels=ClusterData.labels_
    
    figs,axs=plt.subplots(1,3,figsize=(15,4))
    
    axs[0].plot(DataPCA[:,0],DataPCA[:,1],'bo',alpha=0.5)
    axs[0].set_xlabel('First Principal Component',fontsize=12)
    axs[0].set_ylabel('Second Principal Component',fontsize=12)
    
    axs[1].plot(Squared.mean(axis=0))
    axs[1].set_xlabel('Number of Neighbors',fontsize=12)
    axs[1].set_ylabel('Minimum Distance',fontsize=12)
    
    MakeClustersPlot(DataPCA,ClusterLabels,axs[2])
    
    for val in axs:
        PlotStyle(val)
        
    return ClusterLabels

def MakeSimplePlotPanel(Headers,Data,kind='Line'):
    """
    Makes a grid of plots from a data frame. One plot for each heder in Headers 
    ----------
    Headers : list
        Header labels of the features to be ploted.
    Data : pandas data frame
        Data to be analized.

    Returns
    -------
    None.

    """
    nHead=len(Headers)
    nrows,ncolumns=GetGridShape(nHead)
    
    subPlotIndexs=[(j,k) for j in range(nrows) for k in range(ncolumns)]
    if kind=='Line':
        fig,axes=plt.subplots(nrows,ncolumns,figsize=(20,20),sharey=True,sharex=True)
    else:
        fig,axes=plt.subplots(nrows,ncolumns,figsize=(20,20),sharex=True)
    
    for val in enumerate(Headers):
        k,hd=val
        if kind=='Line':
            axes[subPlotIndexs[k]].plot(Data[hd],label=hd)
        elif kind=='Histogram':
            axes[subPlotIndexs[k]].hist(Data[hd],label=hd,bins=50)
        else:
            print('T_T')
            
        axes[subPlotIndexs[k]].legend(loc=1)
    
    for inx in subPlotIndexs:
        PlotStyle(axes[inx])
        
def MakeComparisonPanel(DataA,DataB):
    '''
    Compares the similarity between two arrays with the euclidean distance, 
    pearson correlation, cosine similarity and the city block distance. 

    Parameters
    ----------
    DataA : 2D array
        Data set.
    DataB : 2D array
        Data set.

    Returns
    -------
    None.

    '''
    
    SizeA=len(DataA)
    SizeB=len(DataB)

    funcs=[ds.euclidean,ds.correlation,ds.cosine,ds.cityblock]
    Names=['Euclidean Distance','Pearson Correlation','Cosine Similarity','City Block Distance']
    
    figs,axs=plt.subplots(2,4,figsize=(17,6))
    
    for i,fun in enumerate(funcs):
    
        dataM=np.zeros((SizeA,SizeB))
        for k in range(SizeA):
            currentA=DataA[k]
            for j in range(SizeB):
                sim=fun(currentA,DataB[j])
                dataM[k,j]=sim
                
        axs[0,i].imshow(dataM,aspect='auto')
        axs[0,i].set_title(Names[i],fontsize=13)
        axs[0,i].set_xlabel('Cluster Two',fontsize=13)
        axs[0,i].set_ylabel('Cluster One',fontsize=13)
        
        axs[1,i].hist(dataM.ravel(),bins=50)
        axs[1,i].set_xlabel(Names[i],fontsize=13)
        axs[1,i].set_ylabel('Frequency',fontsize=13)
        
        PlotStyle(axs[1,i])
    
def MakeHistogramPanel(Data,Reference):
    '''
    Compares a data set with a simgle sample with the euclidean distance, 
    pearson correlation, cosine similarity and the city block distance. 

    Parameters
    ----------
    Data : 2D array
        Data set.
    Reference : array
        Reference sample.

    Returns
    -------
    None.

    '''
    
    funcs=[ds.euclidean,ds.correlation,ds.cosine,ds.cityblock]
    Names=['Euclidean Distance','Pearson Correlation','Cosine Similarity','City Block Distance']
    
    figs,axs=plt.subplots(1,4,figsize=(15,4))
    
    for i,fun in enumerate(funcs):
        container=[]
        for val in Data:
            container.append(fun(Reference,val))
        axs[i].hist(container,bins=50)
        axs[i].set_xlabel(Names[i],fontsize=13)
        axs[i].set_ylabel('Frequency',fontsize=13)

    for val in axs:
        PlotStyle(val)

def MakeDispersionPlot(DataFrame,ax):
    '''
    Line plot with a shadow between the standar deviation of the data

    Parameters
    ----------
    DataFrame : pandas data frame
        Data to be plotted.
    ax : matplotlib axes object
        axes for the plot.

    Returns
    -------
    None.

    '''
    
    labels=DataFrame.keys()
    xs=np.arange(len(labels))
    
    upperBound=DataFrame.mean()+DataFrame.std()
    lowerBound=DataFrame.mean()-DataFrame.std()
    
    ax.plot(xs,DataFrame.mean(),'k')
    ax.plot(xs,upperBound,'b',alpha=0.5)
    ax.plot(xs,lowerBound,'b',alpha=0.5)
    
    ax.set_xticks(xs)
    ax.set_xticklabels(labels,rotation=85,fontsize=14)
    ax.set_ylabel('Mean Relative Frequency',fontsize=13)
    ax.fill_between(xs,upperBound,lowerBound,color='b',alpha=0.25)
    PlotStyle(ax)

#Wrapper function, makes a panel of MakeDispersionPlot
def MakeDispersionPanel(ReadingFrames):
    
    fig,axes=plt.subplots(2,1,figsize=(15,15))
    
    for axx,df in zip(axes,ReadingFrames):
        MakeDispersionPlot(df,axx)

def MakeCodonComparisonPlot(DataFrame,ax):
    '''
    Makes a comparison plot, positive values in blue and negative values in red. 

    Parameters
    ----------
    DataFrame : Pandas data frame
        Data to be plotted.
    ax : matplotlib axes object 
        Axes for the plot.

    Returns
    -------
    None.

    '''
    
    labels=DataFrame.keys()
    xs=np.arange(len(labels))
    
    for val,sal in zip(xs,DataFrame):
        
        if sal>0:
            ax.bar(val,sal,color='b')
            ax.text(val-0.5,sal+0.05,labels[val],rotation=85,fontsize=12)
        else:
            ax.bar(val,sal,color='r')
            ax.text(val-0.5,sal-0.045,labels[val],rotation=85,fontsize=12)
            
    ax.set_ylim(-0.5,0.5)
    ax.set_xticks([])
    PlotStyle(ax)
    ax.spines['bottom'].set_visible(False)
    ax.set_ylabel('Relative Change',fontsize=13)

#Wrapper function, creates a panel of MakeCodonComparisonPlot
def MakeCodonComparisonPanel(DataA,DataB):
    
    nrows=len(DataA)
    ncolumns=len(DataB)
    
    fig,axes=plt.subplots(nrows,ncolumns,figsize=(15,15))
    
    if nrows==1 or ncolumns==1:
        maxInx=max([nrows,ncolumns])
        subPlotIndexs=[j for j in range(maxInx)]
    else:
        subPlotIndexs=[(j,k) for j in range(nrows) for k in range(ncolumns)]
    
    k=0
    
    for val in DataA:
        for sal in DataB:
            cInx=subPlotIndexs[k]
            if list==type(val):
                MakeCodonComparisonPlot(val-sal.mean(),axes[cInx])
            elif list==type(sal):
                MakeCodonComparisonPlot(val.mean()-sal,axes[cInx])
            else:
                MakeCodonComparisonPlot(val.mean()-sal.mean(),axes[cInx])
            k=k+1

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
    
    ax.plot(DataPCA[:,0],DataPCA[:,1],'bo',alpha=0.25)
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

def MakeSurveillancePlot(Data,ax,Index,outliers=20):
    '''
    Makes a simple scatter plot froma PCA projection where data points at 
    Index are gray and outlier data points are red

    Parameters
    ----------
    Data : array-like
        Data to be ploted.
    ax : matpotlib axes object
        axes to be used in the plot.
    Index : array
        Location of the samples in Data to be used for the PCA projection.
    outliers : int, optional
        Number of outlier samples to be added in the plot. The default is 20.

    Returns
    -------
    None.

    '''
    
    localIndex=[k for k in range(len(Data))]
    srvIndex=[val for val in localIndex if val not in Index]
    np.random.shuffle(srvIndex)
    
    refData=Data[Index]
    outData=Data[list(srvIndex[0:outliers])]
    
    MethodPCA=PCA(n_components=2)
    MethodPCA.fit(Data[Index+list(srvIndex[0:outliers])])
    refDataPCA=MethodPCA.transform(refData)
    outDataPCA=MethodPCA.transform(outData)
    
    ax.plot(refDataPCA[:,0],refDataPCA[:,1],'o',color='gray',alpha=0.5)
    ax.plot(outDataPCA[:,0],outDataPCA[:,1],'o',color='red',alpha=0.75)
    ax.set_xlabel('First Principal Component',fontsize=12)
    ax.set_ylabel('Second Principal Component',fontsize=12)
    
#Wrapper function that creates a panel of MakeSurveillancePlot
def MakeSurveillancePanel(Data,Index,outliers=20):
    
    nrows,ncolumns=GetGridShape(len(Data))
    
    fig,axes=plt.subplots(nrows,ncolumns,figsize=(15,15))
    Titles=['1-mer','1-mer+2-mer','1-mer+2-mer+3-mer','1-mer+2-mer+3-mer+4-mer','1-mer+2-mer+3-mer+4-mer+5-mer']
    
    if nrows==1 or ncolumns==1:
        maxInx=max([nrows,ncolumns])
        subPlotIndexs=[j for j in range(maxInx)]
    else:
        subPlotIndexs=[(j,k) for j in range(nrows) for k in range(ncolumns)]
        
    for k ,dset in enumerate(Data):
        cInx=subPlotIndexs[k]
        MakeSurveillancePlot(dset,axes[cInx],Index,outliers=outliers)
        axes[cInx].set_title(Titles[k],fontsize=12)
        
    for val in subPlotIndexs:
        PlotStyle(axes[val])
        

###############################################################################
# Sequence/strings utility functions
###############################################################################

#Wrapper function to find a term in a text
def FindTermInText(Text,TermList):
    responce=-1
    for k,val in enumerate(TermList):
        if re.match('(.*)'+val+'(.*)',Text):
            responce=k
            break
    return responce

#Wrapper function to ask if the number of unique elements in a sequence is equal to 4 
def CanonicalAlphabetQ(sequence):
    if len(np.unique(sequence.seq))==4:
        return 1
    else:
        return 0

###############################################################################
# Sequence analysis functions
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

#Wrapper function splits an string and returns the unique elements in the list
def SplitAndUnique(String,FragmentSize):
    unique=set(SplitString(String,FragmentSize))
    return list(unique)

def GetSequencesAlphabetIndex(Sequences):
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
    
    canonical=[]
    noncanonical=[]
    for k,val in enumerate(canonicalResponce):
        if val==1:
            canonical.append(k)
        else:
            noncanonical.append(k)
            
    return canonical,noncanonical

def GetNormalizedUniqueElementsPerSequence(Sequence,MaxFragmentSize):
    '''
    Calculates the ratio between the number of unique elements and the sequence
    length at a given FragmentSize util MaxFragmentSize

    Parameters
    ----------
    Sequence : String
        Data.
    MaxFragmentSize : int
        Max Size of the fragment to be analized.

    Returns
    -------
    Container : list
        Ratio of each fragment size.

    '''
    
    Container=[]
    for k in range(0,MaxFragmentSize):
        cString=SplitString(Sequence,k)
        if len(cString)==0:
            Container.append(1)
        else:
            unique=np.unique(cString)
            Container.append(len(unique)/len(cString))
            
    return Container

#Wrapper function, iterates GetNormalizedUniqueElementsPerSequence throughout a
#list of sequences
def GetNormalizedUniqueElements(Sequences,MaxFragmentSize):
        
    localPool=mp.Pool(MaxCPUCount)
    normUnique=localPool.starmap(GetNormalizedUniqueElementsPerSequence, [(val,MaxFragmentSize )for val in Sequences])
    localPool.close()
    
    return np.array(normUnique)

def GetUniqueByBatch(ListOfElements):
    '''
    Auxiliary function for GetUniqueParallel. Reshapes a list of elements and 
    parllelizes the

    Parameters
    ----------
    ListOfElements : list of elements
        List of unique elements of a series of sequences.

    Returns
    -------
    list
        lis of lists of unique elements.

    '''
    
    MaxBatchSize=150000
    
    if len(ListOfElements)%MaxBatchSize!=0:
        toAdd=MaxBatchSize-len(ListOfElements)%MaxBatchSize
        toAppend=ListOfElements[0]
        
        for k in range(toAdd):
            ListOfElements.append(toAppend)
            
    nshape=int(len(ListOfElements)/MaxBatchSize)
    SquaredUniques=np.array(ListOfElements).reshape((nshape,MaxBatchSize))
        
    setPool=mp.Pool(MaxCPUCount)
    smallSets=setPool.map(set,[val for val in SquaredUniques])
    setPool.close()
    
    return [list(val) for val in smallSets]

def GetUniqueParallel(DataBase,FragmentSize):
    '''
    Each element in the database is divided into fragments of size FragmentSize
    and the unique elements in the datra base are returned

    Parameters
    ----------
    DataBase : list,array
        Contains the sequences to be analyzed.
    FragmentSize : int
        Size of the fragment.

    Returns
    -------
    array
        Unique elements of FragmentSize size present in the data base.

    '''
    
    splitPool=mp.Pool(MaxCPUCount)
    uniques=splitPool.starmap(SplitAndUnique, [(val,FragmentSize )for val in DataBase])
    splitPool.close()
    
    FlatUniquesBack=Flatten(uniques)
    FlatUniquesFoward=[]
    
    for k in range(100):
        FlatUniquesFoward=GetUniqueByBatch(FlatUniquesBack)
        FlatUniquesFoward=Flatten(FlatUniquesFoward)
        
        if len(FlatUniquesBack)-len(FlatUniquesFoward)<450000:
            break
        else:
            FlatUniquesBack=FlatUniquesFoward
    
    return np.unique(FlatUniquesFoward)

###############################################################################
# Data set generating functions
###############################################################################

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

def CountUniqueElements(UniqueElements,String,Processed=False):
    '''
    Calculates the frequency of the unique elements in a splited or 
    processed string. Returns a list with the frequency of the 
    unique elements. 
    
    Parameters
    ----------
    UniqueElements : array,list
        Elements to be analized.
    String : strting
        Sequence data.
    Processed : bool, optional
        Controls if the sring is already splitted or not. The default is False.

    Returns
    -------
    localCounter : array
        Normalized frequency of each unique fragment.

    '''
    
    nUnique=len(UniqueElements)
    localCounter=[0 for k in range(nUnique)]
    
    if Processed:
        ProcessedString=String
    else:
        ProcessedString=SplitString(String,len(UniqueElements[0]))
        
    nSeq=len(ProcessedString)
    UniqueDictionary=UniqueToDictionary(UniqueElements)
    
    for val in ProcessedString:
        try:
            localPosition=UniqueDictionary[val]
            localCounter[localPosition]=localCounter[localPosition]+1
        except KeyError:
            pass
        
    localCounter=[val/nSeq for val in localCounter]
    
    return localCounter

def CountUniqueElementsByBlock(Sequences,UniqueElementsBlock,config=False):
    '''
    

    Parameters
    ----------
    Sequences : list, array
        Data set.
    UniqueElementsBlock : list,array
        Unique element collection of different fragment size.
    config : bool, optional
        Controls if the sring is already splitted or not. The default is False.

    Returns
    -------
    Container : array
        Contains the frequeny of each unique element.

    '''
    
    Container=np.array([[],[]])
    
    for k,block in enumerate(UniqueElementsBlock):
        
        countPool=mp.Pool(MaxCPUCount)
        if config:
            currentCounts=countPool.starmap(CountUniqueElements, [(block,val,True )for val in Sequences])
        else:    
            currentCounts=countPool.starmap(CountUniqueElements, [(block,val )for val in Sequences])
        countPool.close()
        
        if k==0:
            Container=np.array(currentCounts)
        else:
            Container=np.hstack((Container,currentCounts))
            
    return Container

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
# Global definitions
###############################################################################

GlobalDirectory=r"/media/tavoglc/storage/storage/LocalData/"
seqDataDir=GlobalDirectory+'sequencessarscov2.fasta'

MaxCPUCount=int(0.6*mp.cpu_count())

Countries=['AUS','China','USA','Pakistan','Japan','Italy','Taiwan','Nepal',
           'Sweden','Hong Kong','Viet Nam','Brazil','Spain','Colombia','Peru',
           'Germany','Israel','Iran','South Korea','France','South Africa',
           'Turkey','India','Greece','Sri Lanka','Malaysia','Czech Republic',
           'Netherlands','Kazakhstan','Thailand','Serbia','Bangladesh','Poland',
           'Tunisia','Jamaica','Guam','Uruguay','Morocco','Russia','Kenya',
           'Nigeria','Egypt','Timor-Leste','Saudi Arabia','Chile','New Zealand',
           'Bahrain','Georgia','Belgium','Lebanon','Mexico','Zambia','Jordan',
           'Belize','Guatemala','Sierra Leone','Ghana','Venezuela','Denmark',
           'Philippines','Puerto Rico','United Kingdom','Malta','Romania',
           'Iraq','Ecuador','West Bank','Canada','Cambodia','Mali','Myanmar']

###############################################################################
# Data Processing
###############################################################################

seqData=GetSeqs(seqDataDir)

descriptionFilteredSeqs=[]

for val in seqData:
    if re.match('(.*)complete genome(.*)',val.description):
        descriptionFilteredSeqs.append(val)

seqLenghts=[len(val.seq) for val in descriptionFilteredSeqs]

plt.figure()
plt.plot(seqLenghts)
ax=plt.gca()
ax.set_xlabel('Sequences',fontsize=12)
ax.set_ylabel('Sequence Size',fontsize=12)
PlotStyle(ax)


FilteredSeqs=[val for val in descriptionFilteredSeqs if len(val.seq)>10000]

filteredLenghts=[len(val.seq) for val in FilteredSeqs]

plt.figure()
plt.hist(filteredLenghts,bins=50)
ax=plt.gca()
ax.set_xlabel('Sequence Size',fontsize=12)
ax.set_ylabel('Frequency',fontsize=12)
PlotStyle(ax)

###############################################################################
# Data selection
###############################################################################

CanonicalIndex,NonCanonicalIndex=GetSequencesAlphabetIndex(FilteredSeqs)
CanonicalSeqs=[FilteredSeqs[val] for val in CanonicalIndex]
NonCanonicalSeqs=[FilteredSeqs[val] for val in NonCanonicalIndex]

###############################################################################
# Normalized number ofd unique elements
###############################################################################

UniqueRatio=GetNormalizedUniqueElements(CanonicalSeqs,13)

plt.figure()
plt.plot(UniqueRatio.mean(axis=0))
ax=plt.gca()
ax.set_xlabel('K-mer Size',fontsize=12)
ax.set_ylabel('Normalized Unique Elements',fontsize=12)
PlotStyle(ax)

###############################################################################
# Bounds of the k-mers
###############################################################################

UniqueElementsBlock=[GetUniqueParallel(CanonicalSeqs,k) for k in range(1,7)]
Lenghts=[len(val) for val in UniqueElementsBlock]
Lenghts=np.cumsum(Lenghts)

###############################################################################
# MinMax Normalization
###############################################################################

Freqs=CountUniqueElementsByBlock(CanonicalSeqs,UniqueElementsBlock)

Scaler=pr.MinMaxScaler()
Scaler.fit(Freqs)
Freqs=Scaler.transform(Freqs)

###############################################################################
# PCA projection
###############################################################################
     
MakePCAPanel([Freqs[:,0:Lenghts[0]],Freqs[:,0:Lenghts[1]],Freqs[:,0:Lenghts[2]],Freqs[:,0:Lenghts[3]],Freqs[:,0:Lenghts[4]]],['1-mer','1-mer+2-mer','1-mer+2-mer+3-mer','1-mer+2-mer+3-mer+4-mer','1-mer+2-mer+3-mer+4-mer+5-mer'])

###############################################################################
# Resulting Culsters
###############################################################################

Labels01=MakePlotPanel(Freqs[:,0:Lenghts[0]],[0.025,5])
Labels02=MakePlotPanel(Freqs[:,0:Lenghts[1]],[0.025,5])
Labels03=MakePlotPanel(Freqs[:,0:Lenghts[2]],[0.1,5])
Labels04=MakePlotPanel(Freqs[:,0:Lenghts[3]],[0.15,5])
Labels05=MakePlotPanel(Freqs[:,0:Lenghts[4]],[0.2,5])
Labels06=MakePlotPanel(Freqs,[0.15,5])

###############################################################################
# sequence characteristics data set
###############################################################################

DataSet0=pd.DataFrame()
DataSet0['ids']=[val.id for val in CanonicalSeqs]
DataSet0['size']=[len(val.seq) for val in CanonicalSeqs]

locs=[]

for val in CanonicalSeqs:
    disc=FindTermInText(val.description,Countries)
    if disc==-1:
        locs.append('NonStandard')
    else:
        locs.append(Countries[disc])

DataSet0['location']=locs
DataSet0['cluster04']=Labels04
DataSet0['cluster05']=Labels05

FeatureName=Flatten(map(list,UniqueElementsBlock))

###############################################################################
# Data characteristics
###############################################################################

plt.figure(figsize=(15,5))

dmean=DataSet0.groupby('location').mean()['size']
dmin=DataSet0.groupby('location').min()['size']
dmax=DataSet0.groupby('location').max()['size']
labels=dmean.keys()
xs=np.arange(len(labels))

plt.plot(xs,dmean,label='Mean')
plt.plot(xs,dmin,label='Min')
plt.plot(xs,dmax,label='Max')
ax=plt.gca()
ax.set_xticks(xs)
ax.set_xticklabels(labels,rotation=85,fontsize=14)
ax.legend()
ax.set_xlabel('Country',fontsize=13)
ax.set_ylabel('Sequence Size',fontsize=13)
PlotStyle(ax)


MakeSimplePlotFromDF(DataSet0['location'].value_counts(),kind='Bar')

MakeSimplePlotFromDF(DataSet0['location'].value_counts()[2::],kind='Bar')

for val in set(Labels05):
    
    data=DataSet0[DataSet0['cluster05']==val]['location'].value_counts()
    MakeSimplePlotFromDF(data,kind='Bar')
    
###############################################################################
# Cluster comparison
###############################################################################

Cluster051Index=[k for k in range(len(Labels05)) if Labels05[k]==1]
Cluster052Index=[k for k in range(len(Labels05)) if Labels05[k]==3]

cluster051=np.array([Freqs[val,0:Lenghts[4]] for val in Cluster051Index])
cluster052=np.array([Freqs[val,0:Lenghts[4]] for val in Cluster052Index])

plt.figure()
plt.hist(DataSet0.iloc[Cluster051Index]['size'],bins=25)
ax=plt.gca()
ax.set_xlabel('Sequence Size',fontsize=13)
ax.set_ylabel('Frequency',fontsize=13)
ax.set_title('Cluster One',fontsize=13)
PlotStyle(ax)

plt.figure()
plt.hist(DataSet0.iloc[Cluster052Index]['size'],bins=25)
ax=plt.gca()
ax.set_xlabel('Sequence Size',fontsize=13)
ax.set_ylabel('Frequency',fontsize=13)
ax.set_title('Cluster Two',fontsize=13)
PlotStyle(ax)

MakeComparisonPanel(cluster051,cluster052)

###############################################################################
# Reference comparison
###############################################################################

ReferenceLoc=[k for k in range(len(CanonicalSeqs)) if re.match('(.*)MN908947.3(.*)',CanonicalSeqs[k].id)][0]
ReferenceSeq=CanonicalSeqs[ReferenceLoc]
ReferenceFreqs=Freqs[ReferenceLoc][0:Lenghts[4]]

MakeHistogramPanel(cluster051,ReferenceFreqs)
MakeHistogramPanel(cluster052,ReferenceFreqs)

###############################################################################
# Pseudo CUB 
###############################################################################

ReferenceTriplets=ReferenceFreqs[Lenghts[1]:Lenghts[2]]
Cluster051Triplets=cluster051[:,Lenghts[1]:Lenghts[2]]
Cluster052Triplets=cluster052[:,Lenghts[1]:Lenghts[2]]
TripletsNames=UniqueElementsBlock[2]

DFC51=pd.DataFrame()
DFC52=pd.DataFrame()
    
for k,val in enumerate(TripletsNames):
    DFC51[val]=Cluster051Triplets[:,k] 
    DFC52[val]=Cluster052Triplets[:,k] 
    
MakeDispersionPanel([DFC51,DFC52])

plt.figure(figsize=(15,7))
ax=plt.gca()
MakeCodonComparisonPlot(DFC51.mean()-DFC52.mean(),ax)

MakeCodonComparisonPanel([DFC51,DFC52],[list(ReferenceTriplets)])

###############################################################################
# Survellance
###############################################################################
        
MakeSurveillancePanel([Freqs[:,0:Lenghts[0]],Freqs[:,0:Lenghts[1]],Freqs[:,0:Lenghts[2]],Freqs[:,0:Lenghts[3]],Freqs[:,0:Lenghts[4]]],Cluster052Index,outliers=2)

MakeSurveillancePanel([Freqs[:,0:Lenghts[0]],Freqs[:,0:Lenghts[1]],Freqs[:,0:Lenghts[2]],Freqs[:,0:Lenghts[3]],Freqs[:,0:Lenghts[4]]],Cluster052Index,outliers=20)

MakeSurveillancePanel([Freqs[:,0:Lenghts[0]],Freqs[:,0:Lenghts[1]],Freqs[:,0:Lenghts[2]],Freqs[:,0:Lenghts[3]],Freqs[:,0:Lenghts[4]]],Cluster052Index,outliers=200)

MakeSurveillancePanel([Freqs[:,0:Lenghts[0]],Freqs[:,0:Lenghts[1]],Freqs[:,0:Lenghts[2]],Freqs[:,0:Lenghts[3]],Freqs[:,0:Lenghts[4]]],Cluster052Index,outliers=2000)
