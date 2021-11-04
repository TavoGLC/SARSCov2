#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Variational autoencoder of SARS Cov 2 Kmer data.

@author: tavoglc
"""
###############################################################################
# Loading packages 
###############################################################################

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from keras import backend as K 

from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input,Activation,Dense,Layer,BatchNormalization

###############################################################################
# Loading packages 
###############################################################################

globalSeed=768

from numpy.random import seed 
seed(globalSeed)

tf.compat.v1.set_random_seed(globalSeed)

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
    
###############################################################################
#Keras Dense Coder. 
###############################################################################

def MakeDenseCoder(InputShape,Units,Latent,UpSampling=False):
    '''
    Parameters
    ----------
    InputShape : tuple
        Data shape.
    Units : list
        List with the number of dense units per layer.
    Latent : int
        Size of the latent space.
    UpSampling : bool, optional
        Controls the behaviour of the function, False returns the encoder while True returns the decoder. 
        The default is False.

    Returns
    -------
    InputFunction : Keras Model input function
        Input Used to create the coder.
    localCoder : Keras Model Object
        Keras model.

    '''
    Units.append(Latent)
    
    if UpSampling:
        denseUnits=Units[::-1]
        Name="Decoder"
    else:
        denseUnits=Units
        Name="Encoder"
    
    InputFunction=Input(shape=InputShape)
    nUnits=len(denseUnits)
    X=Dense(denseUnits[0],use_bias=False)(InputFunction)
    X=BatchNormalization()(X)
    X=Activation('relu')(X)
    
    for k in range(1,nUnits-1):
        X=Dense(denseUnits[k],use_bias=False)(X)
        X=BatchNormalization()(X)
        X=Activation('relu')(X)
    
    X=Dense(denseUnits[-1],use_bias=False)(X)
    X=BatchNormalization()(X)
    
    if UpSampling:
        Output=Activation('sigmoid')(X)
        localCoder=Model(inputs=InputFunction,outputs=Output,name=Name)
    else:    
        Output=Activation('relu')(X)
        localCoder=Model(inputs=InputFunction,outputs=Output,name=Name)
    
    return InputFunction,localCoder


###############################################################################
#Keras Dense Autoencoder
###############################################################################

def MakeDenseAutoencoder(InputShape,Units,Latent):
    
    InputEncoder,Encoder=MakeDenseCoder(InputShape,Units,Latent)
    InputDecoder,Decoder=MakeDenseCoder((Latent,),Units,Latent,UpSampling=True)
    AEoutput=Decoder(Encoder(InputEncoder))
    AE=Model(inputs=InputEncoder,outputs=AEoutput)
    
    return Encoder,Decoder,AE

###############################################################################
# Keras Utilituy functions
###############################################################################
(0.1*(2/340))
class KLDivergenceLayer(Layer):
    '''
    Custom layer to add the divergence loss to the final model
    '''
    
    def _init_(self,*args,**kwargs):
        self.is_placeholder=True
        super(KLDivergenceLayer,self)._init_(*args,**kwargs)
        
    def call(self,inputs):
        
        Mu,LogSigma=inputs
        klbatch=-0.5*(0.075*(2/340))*K.sum(1+LogSigma-K.square(Mu)-K.exp(LogSigma),axis=-1)
        self.add_loss(K.mean(klbatch),inputs=inputs)
        
        return inputs

class Sampling(Layer):
    '''
    Custom Layer for sampling the latent space
    '''
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def get_config(self):
        config = {}
        base_config = super().get_config()
        return {**base_config, **config}
    
    @tf.autograph.experimental.do_not_convert   
    def call(self,inputs,**kwargs):
        
        Mu,LogSigma=inputs
        batch=tf.shape(Mu)[0]
        dim=tf.shape(Mu)[1]
        epsilon=K.random_normal(shape=(batch,dim))

        return Mu+(K.exp(0.5*LogSigma))*epsilon
    
def MakeVariationalNetwork(Latent):
    
    InputFunction=Input(shape=(Latent,))
    Mu=Dense(Latent)(InputFunction)
    LogSigma=Dense(Latent)(InputFunction)
    Mu,LogSigma=KLDivergenceLayer()([Mu,LogSigma])
    Output=Sampling()([Mu,LogSigma])
    variationalBottleneck=Model(inputs=InputFunction,outputs=Output)
    
    return InputFunction,variationalBottleneck

###############################################################################
# Keras Dense Variational Autoencoder
###############################################################################

def MakeVariationalDenseAutoencoder(InputShape,Units,Latent):
    
    InputEncoder,Encoder=MakeDenseCoder(InputShape,Units,Latent)
    InputVAE,VAE=MakeVariationalNetwork(Latent)
    InputDecoder,Decoder=MakeDenseCoder((Latent,),Units,Latent,UpSampling=True)
    
    VAEencoderOutput=VAE(Encoder(InputEncoder))
    VAEencoder=Model(inputs=InputEncoder,outputs=VAEencoderOutput)
    
    VAEOutput=Decoder(VAEencoder(InputEncoder))
    VAEAE=Model(inputs=InputEncoder,outputs=VAEOutput)
    
    return VAEencoder,Decoder,VAEAE

###############################################################################
# Data loading and splitting
###############################################################################

GlobalDirectory=r"/media/tavoglc/Datasets/DABE/LifeSciences/COVIDSeqs/sep2021"
KmerDataDir = GlobalDirectory+'/kmers/kmerdata.csv'

KmerData = pd.read_csv(KmerDataDir,index_col=0)

KmerLabels = [val for val in list(KmerData) if val not in ['ids']]
KmerXData = KmerData[KmerLabels]
KmerLabels =  KmerData['ids']

Xtrain,Xtest,Ytrain,Ytest = train_test_split(KmerXData,KmerLabels,test_size=0.10,train_size=0.90,random_state=23)

###############################################################################
# Autoencoder
###############################################################################

InputShape = Xtrain.shape[1]
Units = [340,240,140,40,20,10,5,3]
Latent = 2

ENC,DEC,AE = MakeDenseAutoencoder(InputShape,Units,Latent)

lr = 0.0025
minlr = 0.00001
epochs = 100
decay = 3*(lr-minlr)/epochs
    
AE.summary()
AE.compile(Adam(lr=lr,decay=decay),loss='mse')
AE.fit(x=Xtrain,y=Xtrain,batch_size=72,epochs=epochs,validation_data=(Xtest,Xtest))
DenseAERepresentation=ENC.predict(KmerXData)

plt.figure(figsize=(8,8))
plt.plot(DenseAERepresentation[:,0],DenseAERepresentation[:,1],'bo',alpha=0.01)
ax=plt.gca()
PlotStyle(ax)

###############################################################################
# VAE Autoencoder
###############################################################################

InputShape = Xtrain.shape[1]
#Units =  [340,300,260,220,180,140,100,60,20,10,5]
Units = [340,240,140,40,20,10,5,3]
Latent = 2

VAEENC,VAEDEC,VAEAE = MakeVariationalDenseAutoencoder(InputShape,Units,Latent)

lr = 0.0025
minlr = 0.00001
epochs = 100
decay = 3*(lr-minlr)/epochs

VAEAE.summary()
VAEAE.compile(Adam(lr=lr,decay=decay),loss='binary_crossentropy')
VAEAE.fit(x=Xtrain,y=Xtrain,batch_size=72,epochs=epochs,validation_data=(Xtest,Xtest))

VariationalRepresentation = VAEENC.predict(KmerXData)

plt.figure(figsize=(8,8))
plt.plot(VariationalRepresentation[:,0],VariationalRepresentation[:,1],'bo',alpha=0.005)
ax=plt.gca()
PlotStyle(ax)

