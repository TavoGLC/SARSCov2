#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
tCopyright (c) 2020 Octavio Gonzalez-Lugo 
o use, copy, modify, merge, publish, distribute, sublicense, and/or sell
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
import copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import lightgbm  as lgb

from sklearn import preprocessing as pr
from sklearn.model_selection import KFold

###############################################################################
# Data Location
###############################################################################

GlobalDirectory=r"/media/tavoglc/storage/storage/LocalData/"
TrainDataDir=GlobalDirectory+"train.json"
TestDataDir=GlobalDirectory+"test.json"

TrainData=pd.read_json(TrainDataDir,lines=True)
TestData=pd.read_json(TestDataDir,lines=True)

###############################################################################
# Plotting functions
###############################################################################

def PlotStyle(Axes):
    
    """
    General style used in all the plots 
    """
    
    Axes.spines['top'].set_visible(False)
    Axes.spines['bottom'].set_visible(True)
    Axes.spines['left'].set_visible(True)
    Axes.spines['right'].set_visible(False)
    Axes.xaxis.set_tick_params(labelsize=14)
    Axes.yaxis.set_tick_params(labelsize=14)

###############################################################################
# Utility functions
###############################################################################

def ExtractData(Data):
    return [val for val in Data]

def FlattenData(Data):
    container=[]
    for val in Data:
        container=container+ExtractData(val)
    return np.array(container)

def SNTrue(Data,LabelA,LabelB):
    container=[]
    for val,sal in zip(TrainData["reactivity"],TrainData["reactivity_error"]):
        if np.mean(val)/np.mean(sal)>1:
            container.append(1)
        else:
            container.append(0)
    return container

def MinTrue(Data,Label):
    container=[]
    for val in Data[Label]:
        if np.min(val)>-0.5:
            container.append(1)
        else:
            container.append(0)
    return container

def SumTrue(Data,Labels):
    container=[]
    ndata=len(Data)
    for k in range(ndata):
        if Data[Labels].iloc[k].sum()==10:
            container.append(1)
        else:
            container.append(0)
    return container



###############################################################################
# Unique elements functions
###############################################################################

def SplitString(String,ChunkSize):
    '''
    
    Split a string using a sliding window. Split the string on fragments of 
    ChunkSize size.
    
    Parameters
    ----------
    String : string
        Sequence of characters without spaces.
    ChunkSize : int
        Fragment of the string being taken.

    Returns
    -------
    Splitted : list
        List that contains the fragments of the string.

    '''
      
    if ChunkSize==1:
        Splitted=[val for val in String]
    
    else:
        nCharacters=len(String)
        Splitted=[String[k:k+ChunkSize] for k in range(nCharacters-ChunkSize)]
        
    return Splitted

def UniqueToDictionary(UniqueElements):
    '''
    
    Maps a list of unique elements to its integer position on a list
    
    Parameters
    ----------
    UniqueElements : list, array-like
        Contains the unique elements to be indexed.

    Returns
    -------
    localDictionary : python dictionary
        Maps the element to its location on the UniqueElements object.

    '''
    
    localDictionary={}
    nElements=len(UniqueElements)
    
    for k in range(nElements):
        localDictionary[UniqueElements[k]]=k
        
    return localDictionary

def CountUniqueElements(UniqueElements,ProcessedString):
    '''
    Parameters
    ----------
    UniqueElements : list, array-like
        Unique elements to be evaluated.
    ProcessedString : list
        Output of split string, list of equal size strings.

    Returns
    -------
    localCounter : list
        Frequency of each element in UniqueElements that appears 
        at ProcessedString.

    '''
    
    nUnique=len(UniqueElements)
    localCounter=[0 for k in range(nUnique)]
    UniqueDictionary=UniqueToDictionary(UniqueElements)
    
    for val in ProcessedString:
        try:
            localPosition=UniqueDictionary[val]
            localCounter[localPosition]=localCounter[localPosition]+1
        except KeyError:
            pass
    return localCounter

def GetUniqueElements(DataBase,FragmentSize):
    '''
    Obtains the unique elements of size FragmentSize that are present 
    on the DataBase
    
    Parameters
    ----------
    DataBase : list, array-like
        List of strings .
    FragmentSize : int
        Size of the fragment to be analyzed.

    Returns
    -------
    array
        Array of the unique elements on the DataBase.

    '''
    
    Container=[]
    counter=0
    
    for val in DataBase:
        
        try:
            newList=set(SplitString(str(val.seq),FragmentSize))
        except AttributeError:
            newList=set(SplitString(val,FragmentSize))
    
        if counter%250==0:
            Container=list(np.unique(Container))
    
        Container=Container+list(newList)
        
    return np.unique(Container)

#Wrapper function to obtain the K unique elements over the DataBase
def GetTokens(DataBase,MaxSize):
    
    container=[]
    for k in range(1,MaxSize+1):
        container.append(GetUniqueElements(DataBase,k))
    return container 
        
###############################################################################
# Feature Engineering Functions
###############################################################################

def GetSequenceFeatures(Sequence,Encodings,Tokens):
    
    frqContainer=[]
    for tok in Tokens:
        fragSize=len(tok[0])
        processEntry=SplitString(Sequence,fragSize)
        localCounts=CountUniqueElements(tok,processEntry)
        frqContainer=frqContainer+localCounts
    
    featContainer=[]
    EncodingDict=UniqueToDictionary(Encodings)
    
    for char in SplitString(Sequence,1):
        
        charContainer=[EncodingDict[char]]+frqContainer
        featContainer.append(charContainer)
        
    return featContainer
            
def MakeSequenceDataSet(Data,Headers,Encodings,Tokens):
    
    ndata=len(Data)
    container=[]
    
    for k in range(ndata):
        Entry=Data.iloc[k]
        localContainer=[]
        for hd,enc,tok in zip(Headers,Encodings,Tokens):
            entryData=GetSequenceFeatures(str(Entry[hd]),enc,tok)
            if len(localContainer)==0:    
                localContainer=entryData
            else:
                localContainer=[sal+val for sal,val in zip(localContainer,entryData)]
        container=container+localContainer
        
    return container
###############################################################################
#Filtering the data set
###############################################################################

ScoredData=pd.DataFrame()

ScoredData["sequence"]=[str(val)[0:68] for val in TrainData["sequence"]]
ScoredData["structure"]=[str(val)[0:68] for val in TrainData["structure"]]
ScoredData["predicted_loop_type"]=[str(val)[0:68] for val in TrainData["predicted_loop_type"]]

DataNames=["reactivity_error","deg_error_Mg_pH10","deg_error_pH10",
           "deg_error_Mg_50C","deg_error_50C","reactivity","deg_Mg_pH10",
           "deg_pH10","deg_Mg_50C","deg_50C"]

for nme in DataNames:
    ScoredData[nme]=[ExtractData(val)[0:68] for val in TrainData[nme]]
    
ScoredData["reactivitySN"]=SNTrue(ScoredData,"reactivity","reactivity_error")
ScoredData["deg_Mg_pH10SN"]=SNTrue(ScoredData,"deg_Mg_pH10","deg_error_Mg_pH10")
ScoredData["deg_pH10SN"]=SNTrue(ScoredData,"deg_pH10","deg_error_pH10")
ScoredData["deg_Mg_50CSN"]=SNTrue(ScoredData,"deg_Mg_50C","deg_error_Mg_50C")
ScoredData["deg_50CSN"]=SNTrue(ScoredData,"deg_50C","deg_error_50C")

ScoredData["reactivitymin"]=MinTrue(ScoredData,"reactivity")
ScoredData["deg_Mg_pH10min"]=MinTrue(ScoredData,"deg_Mg_pH10")
ScoredData["deg_pH10min"]=MinTrue(ScoredData,"deg_pH10")
ScoredData["deg_Mg_50Cmin"]=MinTrue(ScoredData,"deg_Mg_50C")
ScoredData["deg_50Cmin"]=MinTrue(ScoredData,"deg_50C")

discHeaders=["reactivitymin","deg_Mg_pH10min","deg_pH10min","deg_Mg_50Cmin","deg_50Cmin","reactivitySN","deg_Mg_pH10SN","deg_pH10SN","deg_Mg_50CSN","deg_50CSN"]
ScoredData["valid"]=SumTrue(ScoredData,discHeaders)

ValidData=ScoredData[ScoredData["valid"]==1]

###############################################################################
#Training Data
###############################################################################

maxSize=15
headers=["sequence",'structure','predicted_loop_type']
MainTokens=[GetTokens(np.array(ValidData[val]),4) for val in headers]
Encodings=[val[0] for val in MainTokens]

MainDataSet=np.array(MakeSequenceDataSet(ValidData,headers,Encodings,MainTokens))

ReactivityTarget=FlattenData(ValidData["reactivity"])
MgpH10Target=FlattenData(ValidData["deg_Mg_pH10"])
pH10Target=FlattenData(ValidData["deg_pH10"])
Mg50CTarget=FlattenData(ValidData["deg_Mg_50C"])
deg50CTarget=FlattenData(ValidData["deg_50C"])


###############################################################################
# Model Generation
###############################################################################
    
def CheckHyperparameters(Parameters):
    """
    Parameters
    ----------
    Parameters : list,array
        contains the hyperparameter values .

    Returns
    -------
    Parameters : list,array
        mantains the hyperparameters between certain boundaries.

    """
    Parameters=list(Parameters)
    Bounds=[[1,250],[1,250],[5,100],[0,1]]
    
    for k in range(len(Parameters)):
        if Parameters[k]<Bounds[k][0] or Parameters[k]>Bounds[k][1]:
            Parameters[k]=np.mean(Bounds[k])
            
    Parameters[0]=np.int(Parameters[0])
    Parameters[1]=np.int(Parameters[1])
    Parameters[2]=np.int(Parameters[2])
            
    return Parameters

#Wrapper funtion to format the hyperparamters 
def FormatHyperparameters(Parameters):
    hypNames=["num_leaves","max_depth","min_data_in_leaf","feature_fraction"]
    hyp={}
    for name,val in zip(hypNames,Parameters):
        hyp[name]=val
        
    hyp['objetive']='root_mean_squared_error'
    hyp['num_iterations']=200
    hyp["n_jobs"]=-2
    hyp["seed"]=10
    return hyp

#Wrapper funtion to train the model
def TrainModel(Xtrain,Ytrain,Xtest,Ytest,HyperParams):
    
    localScaler=pr.MinMaxScaler()
    localScaler.fit(Xtrain)
    Xtrain=localScaler.transform(Xtrain)
    Xtest=localScaler.transform(Xtest)
    
    trainData=lgb.Dataset(Xtrain,label=Ytrain)
    testData=Xtest
    
    localModel=lgb.train(HyperParams,trainData)
    localPred=localModel.predict(testData)
    
    return ((Ytest - localPred) ** 2).mean()

#Wrapper function for KFold crossvalidation 
def TrainKFoldCVModel(XData,YData,HyperParams,splits):
    
    fitness=[]
    cKF=KFold(n_splits=splits,shuffle=True,random_state=12)
    
    for trainI,testI in cKF.split(XData,YData):
        Xtrain,Xtest=XData[trainI],XData[testI]
        Ytrain,Ytest=YData[trainI],YData[testI]
        fitness.append(TrainModel(Xtrain,Ytrain,Xtest,Ytest,HyperParams))
        
    return np.mean(fitness)

###############################################################################
#Particle Swarm Optimization
###############################################################################

def UpdateSwarm(Swarm,Velocity,BestIndividual,BestGlobal,InertiaC,SocialC,CognitiveC):
    """
    Parameters
    ----------
    Swarm : list,array
        Swarm particles positions.
    Velocity : list,array
        Swarm velocities.
    BestIndividual : list,array
        Best performance for each particle.
    BestGlobal : list, array
        Global best particle.
    InertiaC : float
        Inertia constant.
    SocialC : float
        Social constant.
    CognitiveC : float
        Cognitive constant.

    Returns
    -------
    newSwarm : list
        updated swarm positions.
    velocity : list
        swarm velocity.

    """
    newSwarm=copy.deepcopy(Swarm)
    velocity=[]
    
    for k in range(len(newSwarm)):
        inertia=InertiaC*np.asarray(Velocity[k])
        social=SocialC*np.random.random()*(np.asarray(BestGlobal)-np.asarray(newSwarm[k]))
        cognitive=CognitiveC*np.random.random()*(np.asarray(BestIndividual[k])-np.asarray(newSwarm[k]))
        vel=inertia+social+cognitive
        velocity.append(vel)
        newSwarm[k]=newSwarm[k]+vel
        
    return newSwarm,velocity

def EvaluateSwarmFitness(XData,YData,Swarm,Splits):
    """
    Parameters
    ----------
    XData : array
        Xdata.
    YData : array
        Ydata.
    Swarm : list,array
        swarm particle positions.
    Splits : int
        number of splits for kfold.

    Returns
    -------
    swarm : list,array
        swarm positions.
    fitness : list
        particle fitness.

    """
    fitness=[]
    swarm=[]
    
    for part in Swarm:
        cPart=CheckHyperparameters(part)
        cHyp=FormatHyperparameters(cPart)
        fitn=TrainKFoldCVModel(XData,YData,cHyp,Splits)
        swarm.append(cPart)
        fitness.append(fitn)
        
    return swarm,fitness

def KFoldCVPSO(XData,YData,Splits,SwarmSize,Iterations,Inertia=0.5,Social=0.25,Cognitive=0.25):
    """
    Parameters
    ----------
    XData : array
        Train Data.
    YData : array
        Train labels.
    Splits : int
        number of splits for k fold.
    SwarmSize : int
        number of particles in the swarm.
    Iterations : int
        Iterations for PSO.
    Inertia : float, optional
        Inertia Constant. The default is 0.5.
    Social : float, optional
        Social Constant. The default is 0.25.
    Cognitive : float, optional
        Cognitive Constant. The default is 0.25.

    Returns
    -------
    Swarm : list
        Optimized hyperparameters.
    loopFitness : list
        Performance of each hyperparameter combination.

    """

    Swarm=np.random.random((SwarmSize,4))
    Velocity=np.random.random((SwarmSize,4))
    
    bestSwarm,bestFitness=EvaluateSwarmFitness(XData,YData,Swarm,Splits)
    
    bestGFitness=np.min(bestFitness)
    bestGPart=bestSwarm[np.argmin(bestFitness)]
    
    for k in range(Iterations):
        
        Swarm,Velocity=UpdateSwarm(Swarm,Velocity,bestSwarm,bestGPart,Inertia,Social,Cognitive)
        Swarm,loopFitness=EvaluateSwarmFitness(XData,YData,Swarm,Splits)
        
        if np.min(loopFitness)<bestGFitness:
            bestGFitness=np.min(loopFitness)
            bestGPart=Swarm[np.argmin(loopFitness)]
        
        for k in range(SwarmSize):
            if loopFitness[k]<bestFitness[k]:
                bestFitness[k]=loopFitness[k]
                bestSwarm[k]=Swarm[k]
        
    return Swarm,loopFitness

###############################################################################
#Particle Swarm Optimization
###############################################################################

maxIterations=5
particles=5
folds=6

paramsR,fitness=KFoldCVPSO(MainDataSet,ReactivityTarget,folds,particles,maxIterations)

plt.figure(figsize=(15,6))
plt.bar(np.arange(len(fitness)),fitness)
plt.xlabel('Models',fontsize=14)
plt.ylabel('% Improvement',fontsize=14)
ax=plt.gca()
PlotStyle(ax)

paramsMP,fitness=KFoldCVPSO(MainDataSet,MgpH10Target,folds,particles,maxIterations)

plt.figure(figsize=(15,6))
plt.bar(np.arange(len(fitness)),fitness)
plt.xlabel('Models',fontsize=14)
plt.ylabel('% Improvement',fontsize=14)
ax=plt.gca()
PlotStyle(ax)

paramsP,fitness=KFoldCVPSO(MainDataSet,pH10Target,folds,particles,maxIterations)

plt.figure(figsize=(15,6))
plt.bar(np.arange(len(fitness)),fitness)
plt.xlabel('Models',fontsize=14)
plt.ylabel('% Improvement',fontsize=14)
ax=plt.gca()
PlotStyle(ax)

paramsMC,fitness=KFoldCVPSO(MainDataSet,Mg50CTarget,folds,particles,maxIterations)

plt.figure(figsize=(15,6))
plt.bar(np.arange(len(fitness)),fitness)
plt.xlabel('Models',fontsize=14)
plt.ylabel('% Improvement',fontsize=14)
ax=plt.gca()
PlotStyle(ax)

paramsC,fitness=KFoldCVPSO(MainDataSet,deg50CTarget,folds,particles,maxIterations)

plt.figure(figsize=(15,6))
plt.bar(np.arange(len(fitness)),fitness)
plt.xlabel('Models',fontsize=14)
plt.ylabel('% Improvement',fontsize=14)
ax=plt.gca()
PlotStyle(ax)

