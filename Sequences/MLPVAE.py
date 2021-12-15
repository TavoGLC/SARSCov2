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

import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from tensorflow import keras

from tensorflow.keras import backend as K 
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Activation, Dense, Layer, BatchNormalization

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
# Keras Utilituy functions
###############################################################################
class KLDivergenceLayer(Layer):
    '''
    Custom KL loss layer
    '''
    def __init__(self,shrinkage,*args,**kwargs):
        self.shrinkage = shrinkage
        self.is_placeholder=True
        super(KLDivergenceLayer,self).__init__(*args,**kwargs)
        
    def call(self,inputs):
        
        Mu,LogSigma=inputs
        klbatch=-0.5*self.shrinkage*K.sum(1+LogSigma-K.square(Mu)-K.exp(LogSigma),axis=-1)
        self.add_loss(K.mean(klbatch),inputs=inputs)
        self.add_metric(klbatch,name='kl_loss',aggregation='mean')
        
        return inputs

class Sampling(Layer):
    '''
    Custom sampling layer
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
    
###############################################################################
#Variational autoencoder bottleneck 
###############################################################################

#Wrapper function, creates a small Functional keras model 
#Bottleneck of the variational autoencoder 
def MakeVariationalNetwork(Latent,shrinkage):
    
    InputFunction=Input(shape=(Latent,))
    Mu=Dense(Latent)(InputFunction)
    LogSigma=Dense(Latent)(InputFunction)
    Mu,LogSigma=KLDivergenceLayer(shrinkage)([Mu,LogSigma])
    Output=Sampling()([Mu,LogSigma])
    variationalBottleneck=Model(inputs=InputFunction,outputs=Output)
    
    return InputFunction,variationalBottleneck


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
# Keras Dense Variational Autoencoder
###############################################################################

def MakeVariationalDenseAutoencoder(InputShape,Units,Latent,shrinkage):
    
    InputEncoder,Encoder=MakeDenseCoder(InputShape,Units,Latent)
    InputVAE,VAE=MakeVariationalNetwork(Latent,shrinkage)
    InputDecoder,Decoder=MakeDenseCoder((Latent,),Units,Latent,UpSampling=True)
    
    VAEencoderOutput=VAE(Encoder(InputEncoder))
    VAEencoder=Model(inputs=InputEncoder,outputs=VAEencoderOutput)
    
    VAEOutput=Decoder(VAEencoder(InputEncoder))
    VAEAE=Model(inputs=InputEncoder,outputs=VAEOutput)
    
    return VAEencoder,Decoder,VAEAE

###############################################################################
# Data loading and splitting
###############################################################################

GlobalDirectory=r"/media/tavoglc/storage/storage/datasets/DABE/LifeSciences/COVIDSeqs/nov2021"
KmerDataDir = GlobalDirectory+'/KmerData.csv'

KmerData = pd.read_csv(KmerDataDir)

KmerLabels = [val for val in list(KmerData) if val not in ['id']]
KmerXData = KmerData[KmerLabels]
KmerLabels =  KmerData['id']

Index = np.arange(len(KmerData))
trainIndex,testIndex,_,_ = train_test_split(Index,Index,test_size=0.10,train_size=0.90,random_state=23)


Xtrain = KmerXData.iloc[trainIndex]
Xtest = KmerXData.iloc[testIndex]    

###############################################################################
# VAE Autoencoder
###############################################################################

InputShape = Xtrain.shape[1]
Units = [340,240,140,40,20,10,5,3]
Latent = 2
sh = 0.0001

lr = 0.0025
minlr = 0.00001
epochs = 100
decay = 3*(lr-minlr)/epochs

VAEENC,VAEDEC,VAEAE = MakeVariationalDenseAutoencoder(InputShape,Units,Latent,sh)
VAEAE.summary()
VAEAE.compile(Adam(lr=lr,decay=decay),loss='mse')
history = VAEAE.fit(x=Xtrain,y=Xtrain,batch_size=72,epochs=epochs,validation_data=(Xtest,Xtest))
    
VariationalRepresentation = VAEENC.predict(KmerXData)
    
fig = plt.figure(figsize = (16,8))

gs = plt.GridSpec(4,4)
    
ax0 = plt.subplot(gs[0:4,0:2])
ax0.plot(VariationalRepresentation[:,0],VariationalRepresentation[:,1],'bo',alpha=0.005)
ax0.title.set_text('Latent Space (shrinkage = ' + str(sh) +')')
PlotStyle(ax0)
    
ax1 = plt.subplot(gs[0:2,2:4])
ax1.plot(history.history['loss'],'k-',label = 'Loss')
ax1.plot(history.history['val_loss'],'r-',label = 'Validation Loss')
ax1.title.set_text('Binary Cross Entropy loss')
ax1.legend(loc=0)
PlotStyle(ax1)
    
ax2 = plt.subplot(gs[2:4,2:4])
ax2.plot(history.history['kl_loss'],'k-',label = 'KL Loss')
ax2.plot(history.history['val_kl_loss'],'r-',label = 'KL Validation Loss')
ax2.title.set_text(' Kullback–Leibler loss')
ax2.legend(loc=0)
PlotStyle(ax2)
    
plt.tight_layout()

toSave = pd.DataFrame()
toSave['id'] = KmerLabels
toSave['Latent0'] = VariationalRepresentation[:,0]
toSave['Latent1'] = VariationalRepresentation[:,1]
toSave.to_csv(GlobalDirectory+'/VAELatent22.csv',index=False)

plt.tight_layout()
plt.savefig(GlobalDirectory+'/fig'+str(str(datetime.datetime.now().time())[0:5])+'.png')
