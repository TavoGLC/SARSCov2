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
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from tensorflow import keras

from tensorflow.keras.utils import Sequence
from tensorflow.keras import backend as K 
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Activation, Dense,Add,concatenate,LeakyReLU
from tensorflow.keras.layers import Conv2D,Conv2DTranspose, Layer,GlobalAveragePooling2D
from tensorflow.keras.layers import Flatten, Reshape, BatchNormalization,GlobalMaxPooling2D


###############################################################################
# Loading packages 
###############################################################################

globalSeed=768

from numpy.random import seed 
seed(globalSeed)

tf.compat.v1.set_random_seed(globalSeed)

#This piece of code is only used if you have a Nvidia RTX or GTX1660 TI graphics card
#for some reason convolutional layers do not work poperly on those graphics cards 

gpus= tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

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
    

def MakeLatentSpaceWalk(ImageSize,Samples,DecoderModel):
    '''
    Parameters
    ----------
    ImageSize : int
        Size of the original image.
    Samples : int
        number of samples to take.
    DecoderModel : keras trained model
        Decoder part of an autoencoder, performs the reconstruction from
        the bottleneck.

    Returns
    -------
    figureContainer : numpy array
        Contains the decode images froim the latent space walk.

    '''

    figureContainer=np.zeros((ImageSize*Samples,ImageSize*Samples))

    gridx = np.linspace(-5,5,Samples)
    gridy = np.linspace(-5,5,Samples)

    for i,yi in enumerate(gridx):
        for j,xi in enumerate(gridy):
            zsample = np.array([[xi,yi]])
            xDec = DecoderModel.predict(zsample)
            digit = xDec[0].reshape(ImageSize,ImageSize)
            figureContainer[i*ImageSize:(i+1)*ImageSize,j*ImageSize:(j+1)*ImageSize] = digit
            
    return figureContainer
    
###############################################################################
# Keras layers
###############################################################################

class KLDivergenceLayer(Layer):
    '''
    Custom KL loss layer
    '''
    def __init__(self,*args,**kwargs):
        self.annealing = tf.Variable(0.,dtype=tf.float32,trainable = False)
        self.is_placeholder=True
        super(KLDivergenceLayer,self).__init__(*args,**kwargs)
        
    def call(self,inputs):
        
        Mu,LogSigma=inputs
        klbatch=-0.5*self.annealing*K.sum(1+LogSigma-K.square(Mu)-K.exp(LogSigma),axis=-1)
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

class SpatialAttention(Layer):
    '''
    Custom Spatial attention layer
    '''
    
    def __init__(self,size, **kwargs):
        super(SpatialAttention, self).__init__()
        self.size = size
        self.kwargs = kwargs

    def build(self, input_shapes):
        self.conv = Conv2D(filters=1, kernel_size=self.size, strides=1, padding='same')

    def call(self, inputs):
        pooled_channels = tf.concat(
            [tf.math.reduce_max(inputs, axis=3, keepdims=True),
            tf.math.reduce_mean(inputs, axis=3, keepdims=True)],
            axis=3)

        scale = self.conv(pooled_channels)
        scale = tf.math.sigmoid(scale)

        return inputs * scale

class ChannelAttention(Layer):
    
    def __init__(self,**kwargs):
        super(ChannelAttention,self).__init__()
        self.kwargs = kwargs
    
    def get_config(self):
        config = super(ChannelAttention,self).get_config().copy()
        config.update({'ratio':self.ratio})
        return config
    
    def build(self,input_shape):
        channel = input_shape[-1]
        self.dense0 = Dense(channel)
        self.dense1 = Dense(channel)
    
    def call(self,inputs):
        
        channel = inputs.get_shape().as_list()[-1]
        
        avgpool = GlobalAveragePooling2D()(inputs)
        avgpool = Reshape((1,1,channel))(avgpool)
        avgpool = self.dense0(avgpool)
        avgpool = self.dense1(avgpool)
        
        maxpool = GlobalMaxPooling2D()(inputs)
        maxpool = Reshape((1,1,channel))(maxpool)
        maxpool = self.dense0(maxpool)
        maxpool = self.dense1(maxpool)
        
        feature = Add()([avgpool,maxpool])
        feature = Activation('sigmoid')(feature)
        
        return inputs*feature

###############################################################################
#Variational autoencoder bottleneck 
###############################################################################

#Wrapper function, creates a small Functional keras model 
#Bottleneck of the variational autoencoder 
def MakeVariationalNetwork(Latent):
    
    InputFunction=Input(shape=(Latent,))
    Mu=Dense(Latent)(InputFunction)
    LogSigma=Dense(Latent)(InputFunction)
    Mu,LogSigma=KLDivergenceLayer(name='KLDivergence')([Mu,LogSigma])
    Output=Sampling()([Mu,LogSigma])
    variationalBottleneck=Model(inputs=InputFunction,outputs=Output)
    
    return InputFunction,variationalBottleneck

###############################################################################
# Autoencoder utility functions
###############################################################################

def MakeBottleneck(InputShape,Latent,UpSampling=False):
    '''
    Parameters
    ----------
    InputShape : tuple
        input shape of the previous convolutional layer.
    Latent : int
        Dimentionality of the latent space.
    UpSampling : bool, optional
        Controls the sampling behaviour of the network.
        The default is False.

    Returns
    -------
    InputFunction : Keras functional model input
        input of the network.
    localCoder : Keras functional model
        Coder model, transition layer of the bottleneck.

    '''
    
    productUnits = np.product(InputShape)
    Units = [productUnits,productUnits//4,productUnits//16,Latent]
    
    if UpSampling:
        finalUnits = Units[::-1]
        InputFunction = Input(shape=(Latent,))
        X = Dense(finalUnits[0],use_bias=False)(InputFunction)
    
    else:
        finalUnits = Units
        InputFunction = Input(shape=InputShape)
        X = Flatten()(InputFunction)
        X = Dense(finalUnits[0],use_bias=False)(X)
                
    
    X = BatchNormalization()(X)
    X = LeakyReLU()(X)
    
    for k in range(1,len(Units)-1):
        
        X = Dense(finalUnits[k],use_bias=False)(X)
        X = BatchNormalization()(X)
        X = LeakyReLU()(X)
    
    X = Dense(finalUnits[-1],use_bias=False)(X)
    
    if UpSampling:
        X=LeakyReLU()(X)
        Output=Reshape(InputShape)(X)
    else:
        Output=LeakyReLU()(X)
        
    Bottleneck=Model(inputs=InputFunction,outputs=Output)
    
    return InputFunction,Bottleneck


###############################################################################
# Keras Mixer 
#Modified from https://keras.io/examples/vision/convmixer/
###############################################################################
def MakeConvolutionBlock(X, Convolutions):
    
    X = Conv2D(Convolutions, (3, 3), padding='same',use_bias=False)(X)
    X = BatchNormalization()(X)
    X = LeakyReLU()(X)

    return X

def MakeDenseBlock(x, Convolutions,Depth):

    concat_feat= x
    for i in range(Depth):
        x = MakeConvolutionBlock(concat_feat,Convolutions)
        concat_feat=concatenate([concat_feat,x])

    return concat_feat

def SamplingBlock(X,Units,Depth,UpSampling=False):
    
    X = MakeDenseBlock(X,Units,Depth)
    
    if UpSampling:
        X = Conv2DTranspose(Units,(3,3),strides=(2,2),padding='same',use_bias=False)(X)
    else:    
        X = Conv2D(Units,(3,3),strides=(2,2),padding='same',use_bias=False)(X)
    
    X = ChannelAttention()(X)
    X = SpatialAttention(3)(X)
    
    X = BatchNormalization()(X)
    X = LeakyReLU()(X)
    
    return X 
    
def CoderByBlock(InputShape,Units,Depth,UpSampling=False):
    
    if UpSampling:
        Units=Units[::-1]
    else:
        Units=Units
    
    InputFunction = Input(shape=InputShape)
    X = SamplingBlock(InputFunction,Units[0],Depth,UpSampling=UpSampling)
    
    for k in range(1,len(Units)-1):
        
        if Depth-k+1 <= 1:
            blockSize = 2
        else:
            blockSize = Depth-k
        
        X = SamplingBlock(X,Units[k],blockSize,UpSampling=UpSampling)
        
    if UpSampling:
        X = Conv2D(1,(3,3),padding='same',use_bias=False)(X)
        Output = Activation('sigmoid')(X)
    else:
        X = Conv2D(Units[-1],(3,3),padding='same',use_bias=False)(X)
        Output = LeakyReLU()(X)
        
    coderModel = Model(inputs=InputFunction,outputs=Output)
    
    return InputFunction,coderModel

###############################################################################
# Autoencoder Model
###############################################################################

#Wrapper function joins the Coder function and the bottleneck function 
#to create a simple autoencoder
def MakeAutoencoder(CoderFunction,InputShape,Units,BlockSize,**kwargs):
    
    InputEncoder,Encoder=CoderFunction(InputShape,Units,BlockSize,**kwargs)
    #Encoder.summary()
    EncoderOutputShape=Encoder.layers[-1].output_shape
    BottleneckInputShape=EncoderOutputShape[1::]
    InputBottleneck,Bottleneck=MakeBottleneck(BottleneckInputShape,2)
    ConvEncoderOutput=Bottleneck(Encoder(InputEncoder))
    
    ConvEncoder=Model(inputs=InputEncoder,outputs=ConvEncoderOutput)
    
    rInputBottleneck,rBottleneck=MakeBottleneck(BottleneckInputShape,2,UpSampling=True)
    InputDecoder,Decoder=CoderFunction(BottleneckInputShape,Units,BlockSize,UpSampling=True,**kwargs)
    #Decoder.summary()
    ConvDecoderOutput=Decoder(rBottleneck(rInputBottleneck))
    ConvDecoder=Model(inputs=rInputBottleneck,outputs=ConvDecoderOutput)
    
    ConvAEoutput=ConvDecoder(ConvEncoder(InputEncoder))
    ConvAE=Model(inputs=InputEncoder,outputs=ConvAEoutput)
    
    return InputEncoder,InputDecoder,ConvEncoder,ConvDecoder,ConvAE

###############################################################################
# Variational Autoencoder Model
###############################################################################

# Wrapper functon, joins the autoencoder function with the custom variational
#layers to create an autoencoder
def MakeVariationalAutoencoder(CoderFunction,InputShape,Units,BlockSize,**kwargs):
    
    InputEncoder,InputDecoder,ConvEncoder,ConvDecoder,_=MakeAutoencoder(CoderFunction,InputShape,Units,BlockSize,**kwargs)
    
    InputVAE,VAE=MakeVariationalNetwork(2)
    VAEencoderOutput=VAE(ConvEncoder(InputEncoder))
    ConvVAEencoder=Model(inputs=InputEncoder,outputs=VAEencoderOutput)
    
    VAEOutput=ConvDecoder(ConvVAEencoder(InputEncoder))
    ConvVAEAE=Model(inputs=InputEncoder,outputs=VAEOutput)
    
    return InputEncoder,InputDecoder,ConvVAEencoder,ConvDecoder,ConvVAEAE    

###############################################################################
# Keras Data Utilituy functions
###############################################################################
    
class DataSequence(Sequence):
    
    def __init__(self, x_set,batch_size):
        self.x = x_set
        self.batch_size = batch_size
        
    def __len__(self):
        return int(np.ceil(len(self.x)/self.batch_size))
    
    def __data_generation(self, dirList):
         
        X = np.array([np.load(val) for val in dirList])
        X = X.reshape((-1,64,80,1))
        y = X

        return X,y

    def __getitem__(self,idx):
        batch_x = self.x[idx*self.batch_size:(idx+1)*self.batch_size]
        X, y = self.__data_generation(batch_x)
        
        return X,y

class KLAnnealing(keras.callbacks.Callback):

    def __init__(self,position, weigths):
        super().__init__()
        self.position = position
        self.weigths = tf.Variable(weigths,trainable=False,dtype=tf.float32)

    def on_epoch_end(self, epoch,logs=None):
        
        weights = self.model.get_weights()
        weights[self.position] = self.weigths[epoch]
        self.model.set_weights(weights)


def MakeAnnealingWeights(epochs,cycles,scale=1):
    
    pointspercycle = epochs//cycles
    AnnealingWeights = 1*(1/(1+np.exp(-1*np.linspace(-10,10,num=pointspercycle))))
    
    for k in range(cycles-1):
        AnnealingWeights = np.append(AnnealingWeights,1*(1/(1+np.exp(-1*np.linspace(-10,10,num=pointspercycle+1)))))
        
    return scale*AnnealingWeights

###############################################################################
# Data path 
###############################################################################

GlobalDirectory=r"/media/tavoglc/storage/storage/datasets/DABE/LifeSciences/COVIDSeqs/nov2021"
matrixData = GlobalDirectory + '/featuresdata'
fileNames = os.listdir(matrixData)
filePaths = np.array([matrixData+'/'+val for val in fileNames])
ids = [val[0:-4] for val in fileNames]

trainPaths,testPaths,_,_ = train_test_split(filePaths,filePaths,test_size=0.05,random_state=36)


input_shape = (64,80,1)
Arch = [12,24,36,48,36]
Depth = 4

lr = 0.0001
minlr = 0.000001
epochs = 60
batch_size = 128
decay = 2*(lr-minlr)/epochs


_,_,Encoder,Decoder,AE = MakeVariationalAutoencoder(CoderByBlock,input_shape,Arch,Depth)
AE.summary()

AnnealingWeights = MakeAnnealingWeights(epochs,4,scale=0.00001)
KLAposition = [k for k,val in enumerate(AE.get_weights()) if len(val.shape)==0][0]


TrainData = DataSequence(trainPaths,batch_size)
TestData = DataSequence(testPaths,batch_size)
FullData = DataSequence(filePaths,batch_size)

AE.compile(Adam(lr=lr,decay=decay),loss='mse')
history =  AE.fit(TrainData,epochs=epochs,
                  validation_data=TestData,workers=2,
                  callbacks=[KLAnnealing(KLAposition,AnnealingWeights)])
  
VariationalRepresentation = Encoder.predict(FullData)

toSave = pd.DataFrame()
toSave['id'] = ids
toSave['Latent0'] = VariationalRepresentation[:,0]
toSave['Latent1'] = VariationalRepresentation[:,1]
toSave.to_csv(GlobalDirectory+'/ConvVAELatent.csv',index=False)

fig = plt.figure(figsize = (16,8))
gs = plt.GridSpec(4,4)
ax0 = plt.subplot(gs[0:4,0:2])
ax0.plot(VariationalRepresentation[:,0],VariationalRepresentation[:,1],'bo',alpha=0.025)
ax0.title.set_text('Latent Space (shrinkage = '+ str(Arch) +')')
PlotStyle(ax0)
        
ax1 = plt.subplot(gs[0:2,2:4])
ax1.plot(history.history['loss'],'k-',label = 'Loss')
ax1.plot(history.history['val_loss'],'r-',label = 'Validation Loss')
ax1.title.set_text('Reconstruction loss')
ax1.legend(loc=0)
PlotStyle(ax1)
        
ax2 = plt.subplot(gs[2:4,2:4])
ax2.plot(history.history['kl_loss'],'k-',label = 'KL Loss')
ax2.plot(history.history['val_kl_loss'],'r-',label = 'KL Validation Loss')
ax2.title.set_text(' Kullback–Leibler loss')
ax2.legend(loc=0)
PlotStyle(ax2)
        
plt.tight_layout()
plt.savefig(GlobalDirectory+'/fig'+str(str(datetime.datetime.now().time())[0:5])+'.png')
