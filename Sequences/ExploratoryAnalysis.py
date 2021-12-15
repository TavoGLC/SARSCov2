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
# Visualization functions   
###############################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans,DBSCAN

###############################################################################
# Visualization functions   
###############################################################################

def ImageStyle(Axes): 
    """
    Parameters
    ----------
    Axes : Matplotlib axes object
        Applies a general style to the matplotlib object

    Returns
    -------
    None.
    """    
    Axes.spines['top'].set_visible(False)
    Axes.spines['bottom'].set_visible(False)
    Axes.spines['left'].set_visible(False)
    Axes.spines['right'].set_visible(False)
    Axes.set_xticks([])
    Axes.set_yticks([])

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

###############################################################################
# Sampling functions 
###############################################################################

def GetSampleLoc(Sample,boundaries):
  cLoc=0
  for k in range(len(boundaries)-1):
    if Sample>=boundaries[k] and Sample<boundaries[k+1]:
      cLoc=k
      break
      
  return cLoc

def GetEqualizedIndex(Data,bins=100,maxCount=100):
  
  cMin,cMax=np.min(Data),np.max(Data)
  boundaries=np.linspace(cMin,cMax,num=bins+1)
  
  SamplesCount=np.zeros(bins)
  indexContainer = []
  
  index=[k for k in range(len(Data))]
  np.random.shuffle(index)
  
  for val in index:
      dataPoint = Data.iloc[val]
      cLoc=GetSampleLoc(dataPoint,boundaries)
      
      if SamplesCount[cLoc]<=maxCount:
          indexContainer.append(val)
          SamplesCount[cLoc]=SamplesCount[cLoc]+1
      
  return indexContainer

###############################################################################
# Data loading and encoding 
###############################################################################

GlobalDirectory=r"/media/tavoglc/storage/storage/datasets/DABE/LifeSciences/COVIDSeqs/nov2021"
Data = pd.read_csv(GlobalDirectory+'/mergedDF.csv')
KmersData = pd.read_csv(GlobalDirectory+'/KmerData.csv')
Data.fillna(0,inplace=True)
KmersData['id'] = [val[0:-2] for val in KmersData['id']]

KmersData = KmersData.set_index('id')

geoUnique = np.array([str(val) for val in set(Data['SimplifiedGEO'])])
geoToval = dict([(val,sal) for val,sal in zip(geoUnique,np.linspace(0,1,num=geoUnique.size)) ])

Data['geo_encoding'] = [geoToval[str(val)] for val in Data['SimplifiedGEO']]

def GetSimplifiedStrain(pangoLineage):
    
    if pangoLineage[0]=='A' or pangoLineage[0]=='B':
        return pangoLineage
    else:
        return 'Non'

def GetBinaryStrain(pangoLineage):
    
    if pangoLineage[0]=='A' or pangoLineage[0]=='B':
        return pangoLineage[0]
    else:
        return 'Non'

uniquePango = [str(val) for val in set(Data['Pangolin'])]
uniqueSimplified = set([GetSimplifiedStrain(val) for val in uniquePango])

Npango = len(uniquePango)
NsimplifiedPango = len(uniqueSimplified)

pangoToval = dict([(val,sal) for val,sal in zip(uniquePango,np.linspace(0,1,num=Npango)) ])
simpToVal = dict([(val,sal) for val,sal in zip(uniqueSimplified,np.linspace(0,1,num=NsimplifiedPango)) ])

Data['pango_encoding'] = [pangoToval[str(val)] for val in Data['Pangolin']]
Data['simpPango_encoding'] = [simpToVal[GetSimplifiedStrain(str(val))] for val in Data['Pangolin']]

binaryToval = dict([('A',0),('B',0.5),('Non',1)])

Data['pango_binary'] = [GetBinaryStrain(str(val)) for val in Data['Pangolin']]
Data['binary_encoding'] = [binaryToval[str(val)] for val in Data['pango_binary']]

###############################################################################
# Bottlenecks 
###############################################################################

plt.figure(figsize=(16,8))
plt.scatter(Data['PCA_A'],Data['PCA_B'],c='blue',alpha=0.05,label='Katya')
plt.legend(loc=1,frameon=False,fontsize='13')
plt.xlabel('Principal Component A')
plt.ylabel('Principal Component B')
PlotStyle(plt.gca()) 

plt.figure(figsize=(16,8))
plt.scatter(Data['VAE_A'],Data['VAE_B'],c='blue',alpha=0.05,label='Nina')
plt.legend(loc=1,frameon=False,fontsize='13')
plt.xlabel('Latent Dimension A')
plt.ylabel('Latent Dimension B')
PlotStyle(plt.gca()) 

plt.figure(figsize=(16,8))
plt.scatter(Data['ConvVAE_A'],Data['ConvVAE_B'],c='blue',alpha=0.05,label='Masha')
plt.legend(loc=1,frameon=False,fontsize='13')
plt.xlabel('Latent Dimension A')
plt.ylabel('Latent Dimension B')
PlotStyle(plt.gca()) 

###############################################################################
# Geographical encoding
###############################################################################

plt.figure(figsize=(16,8))
plt.scatter(Data['PCA_A'],Data['PCA_B'],c=Data['geo_encoding'],alpha=0.05,cmap='viridis',label='Geographical encoding')
plt.legend(loc=1,frameon=False,fontsize='13')
plt.title('Katya',loc='right')
plt.xlabel('Principal Component A')
plt.ylabel('Principal Component B')
PlotStyle(plt.gca()) 

plt.figure(figsize=(16,8))
plt.scatter(Data['VAE_A'],Data['VAE_B'],c=Data['geo_encoding'],alpha=0.05,cmap='viridis',label='Geographical encoding')
plt.legend(loc=1,frameon=False,fontsize='13')
plt.title('Nina',loc='right')
plt.xlabel('Latent Dimension A')
plt.ylabel('Latent Dimension B')
PlotStyle(plt.gca()) 

plt.figure(figsize=(16,8))
plt.scatter(Data['ConvVAE_A'],Data['ConvVAE_B'],c=Data['geo_encoding'],alpha=0.05,cmap='viridis',label='Geographical encoding')
plt.legend(loc=1,frameon=False,fontsize='13')
plt.title('Masha',loc='right')
plt.xlabel('Latent Dimension A')
plt.ylabel('Latent Dimension B')
PlotStyle(plt.gca()) 

###############################################################################
# Time encoding
###############################################################################

plt.figure(figsize=(16,8))
plt.scatter(Data['PCA_A'],Data['PCA_B'],c=Data['week'],alpha=0.05,cmap='viridis',label='Time encoding')
plt.legend(loc=1,frameon=False,fontsize='13')
plt.title('Katya',loc='right')
plt.xlabel('Principal Component A')
plt.ylabel('Principal Component B')
PlotStyle(plt.gca()) 

plt.figure(figsize=(16,8))
plt.scatter(Data['VAE_A'],Data['VAE_B'],c=Data['week'],alpha=0.05,cmap='viridis',label='Time encoding')
plt.legend(loc=1,frameon=False,fontsize='13')
plt.title('Nina',loc='right')
plt.xlabel('Latent Dimension A')
plt.ylabel('Latent Dimension B')
PlotStyle(plt.gca()) 

plt.figure(figsize=(16,8))
plt.scatter(Data['ConvVAE_A'],Data['ConvVAE_B'],c=Data['week'],alpha=0.05,cmap='viridis',label='Time encoding')
plt.legend(loc=1,frameon=False,fontsize='13')
plt.title('Masha',loc='right')
plt.xlabel('Latent Dimension A')
plt.ylabel('Latent Dimension B')
PlotStyle(plt.gca()) 

###############################################################################
# Time Encoding Katya
###############################################################################

PCAkmeans = KMeans(n_clusters=2, random_state=0).fit(Data[['PCA_A','PCA_B']])
Data['PCA_Labels'] = PCAkmeans.labels_

localColors = ['blue','red']

plt.figure(figsize=(14,10))
gs = plt.GridSpec(4,4)

ax1 = plt.subplot(gs[0:2,0:4])
for k in range(2):
    ax1.scatter(Data[Data['PCA_Labels']==k]['PCA_A'],Data[Data['PCA_Labels']==k]['PCA_B'],c=localColors[k],alpha=0.05)

ax1.set_title('Katya',loc='right')
ax1.set_xlabel('Principal Component A')
ax1.set_ylabel('Principal Component B')

ax3 = plt.subplot(gs[2:4,0])
ax3.hist(Data[Data['PCA_Labels']==0]['month'],bins=10,color='red')
ax3.set_xlabel('Normalized Time (Month)')

ax4 = plt.subplot(gs[2:4,1])
ax4.hist(Data[Data['PCA_Labels']==1]['month'],bins=10,color='blue')
ax4.set_xlabel('Normalized Time (Month)')

ax5 = plt.subplot(gs[2:4,2])
ax5.hist(Data[Data['PCA_Labels']==0]['week'],bins=20,color='red')
ax5.set_xlabel('Normalized Time (week)')

ax6 = plt.subplot(gs[2:4,3])
ax6.hist(Data[Data['PCA_Labels']==1]['week'],bins=20,color='blue')
ax6.set_xlabel('Normalized Time (week)')

[PlotStyle(val) for val in [ax1,ax3,ax4,ax5,ax6]]
plt.tight_layout()

###############################################################################
# Time Encoding Nina
###############################################################################

VAEkmeans =  DBSCAN(eps=0.045, min_samples=200,algorithm='ball_tree').fit(Data[['VAE_A','VAE_B']])
Data['VAE_Labels'] = VAEkmeans.labels_

clustersLabels,counts = np.unique(VAEkmeans.labels_,return_counts=True)
clabels = []

for val,sal in zip(clustersLabels,counts):
    if sal>1000 and val!=-1:
        clabels.append(val)

localColors = [plt.cm.seismic_r(val) for val in np.linspace(0,1,num=len(clabels))]

plt.figure(figsize=(14,10))
gs = plt.GridSpec(4,4)

ax1 = plt.subplot(gs[0:2,0:4])

for k in clabels:
    ax1.scatter(Data[Data['VAE_Labels']==k]['VAE_A'],Data[Data['VAE_Labels']==k]['VAE_B'],c=localColors[k],alpha=0.0125)

ax1.scatter(Data[Data['VAE_Labels']==-1]['VAE_A'],Data[Data['VAE_Labels']==-1]['VAE_B'],c='black',alpha=0.0125)
ax1.set_title('Nina',loc='right')
ax1.set_xlabel('Latent Dimension A')
ax1.set_ylabel('Latent Dimension B')

ax2 = plt.subplot(gs[2,0])
ax2.hist(Data[Data['VAE_Labels']==clabels[0]]['month'],bins=20,color=localColors[0])
ax2.set_xlabel('Normalized Time (Month)')

ax3 = plt.subplot(gs[2,1])
ax3.hist(Data[Data['VAE_Labels']==clabels[1]]['month'],bins=20,color=localColors[1])
ax3.set_xlabel('Normalized Time (Month)')

ax4 = plt.subplot(gs[2,2])
ax4.hist(Data[Data['VAE_Labels']==clabels[2]]['month'],bins=20,color=localColors[2])
ax4.set_xlabel('Normalized Time (Month)')

ax9 = plt.subplot(gs[2,3])
ax9.hist(Data[Data['VAE_Labels']==clabels[3]]['month'],bins=20,color=localColors[3])
ax9.set_xlabel('Normalized Time (Month)')

ax5 = plt.subplot(gs[3,0])
ax5.hist(Data[Data['VAE_Labels']==clabels[0]]['week'],bins=20,color=localColors[0])
ax5.set_xlabel('Normalized Time (week)')

ax6 = plt.subplot(gs[3,1])
ax6.hist(Data[Data['VAE_Labels']==clabels[1]]['week'],bins=20,color=localColors[1])
ax6.set_xlabel('Normalized Time (week)')

ax7 = plt.subplot(gs[3,2])
ax7.hist(Data[Data['VAE_Labels']==clabels[2]]['week'],bins=20,color=localColors[2])
ax7.set_xlabel('Normalized Time (week)')

ax8 = plt.subplot(gs[3,3])
ax8.hist(Data[Data['VAE_Labels']==clabels[3]]['week'],bins=20,color=localColors[3])
ax8.set_xlabel('Normalized Time (week)')

[PlotStyle(val) for val in [ax2,ax1,ax3,ax4,ax5,ax6,ax7,ax8,ax9]]
plt.tight_layout()

fig,axes=plt.subplots(4,4,figsize=(15,15),sharex=True,sharey=True)
for k in range(4):
    for j in range(4):    
        axes[k,j].hist(KmersData['A'].loc[Data[Data['VAE_Labels']==k]['id']],color='red',bins=100,density=True,label='Cluster = '+str(k))
        axes[k,j].hist(KmersData['A'].loc[Data[Data['VAE_Labels']==j]['id']],color='red',bins=100,alpha=0.5,density=True,label='Cluster = '+str(j))
        axes[k,j].set_xlim([0.05,0.15])
        axes[k,j].legend(loc=1)
        PlotStyle(axes[k,j])
fig.suptitle('Adenine shift',x=0.9,y=0.9)

fig,axes=plt.subplots(4,4,figsize=(15,15),sharex=True,sharey=True)
for k in range(4):
    for j in range(4):    
        axes[k,j].hist(KmersData['C'].loc[Data[Data['VAE_Labels']==k]['id']],color='blue',bins=100,density=True,label='Cluster = '+str(k))
        axes[k,j].hist(KmersData['C'].loc[Data[Data['VAE_Labels']==j]['id']],color='blue',bins=100,alpha=0.5,density=True,label='Cluster = '+str(j))
        axes[k,j].set_xlim([0.2,0.35])
        axes[k,j].legend(loc=1)
        PlotStyle(axes[k,j])
fig.suptitle('Cytosine shift',x=0.9,y=0.9)

fig,axes=plt.subplots(4,4,figsize=(15,15),sharex=True,sharey=True)
for k in range(4):
    for j in range(4):    
        axes[k,j].hist(KmersData['G'].loc[Data[Data['VAE_Labels']==k]['id']],color='green',bins=100,density=True,label='Cluster = '+str(k))
        axes[k,j].hist(KmersData['G'].loc[Data[Data['VAE_Labels']==j]['id']],color='green',bins=100,alpha=0.5,density=True,label='Cluster = '+str(j))
        axes[k,j].set_xlim([0.75,0.85])
        axes[k,j].legend(loc=1)
        PlotStyle(axes[k,j])
fig.suptitle('Guanine shift',x=0.9,y=0.9)

fig,axes=plt.subplots(4,4,figsize=(15,15),sharex=True,sharey=True)
for k in range(4):
    for j in range(4):    
        axes[k,j].hist(KmersData['T'].loc[Data[Data['VAE_Labels']==k]['id']],color='black',bins=100,density=True,label='Cluster = '+str(k))
        axes[k,j].hist(KmersData['T'].loc[Data[Data['VAE_Labels']==j]['id']],color='black',bins=100,alpha=0.5,density=True,label='Cluster = '+str(j))
        axes[k,j].set_xlim([0.7,0.8])
        axes[k,j].legend(loc=1)
        PlotStyle(axes[k,j])
fig.suptitle('Thymine/Uracil shift',x=0.9,y=0.9)

###############################################################################
# Time Encoding Masha
###############################################################################

ConvVAEkmeans =  DBSCAN(eps=0.020,min_samples=15,algorithm='ball_tree',metric='euclidean',n_jobs=-2).fit(Data[['ConvVAE_A','ConvVAE_B']])
clustersLabels,counts = np.unique(ConvVAEkmeans.labels_,return_counts=True)
clabels = []

for val,sal in zip(clustersLabels,counts):
    if sal>1000 and val!=-1:
        clabels.append(val)

localColors = [plt.cm.seismic(val) for val in np.linspace(0,1,num=len(clabels))]
Data['ConvVAE_Labels'] = [val if val in clabels else -1 for val in ConvVAEkmeans.labels_]

plt.figure(figsize=(18,10))

gs = plt.GridSpec(2,11)
ax0 = plt.subplot(gs[0,:])

for k,col in zip(clabels,localColors):
    ax0.scatter(Data[Data['ConvVAE_Labels']==k]['ConvVAE_A'],Data[Data['ConvVAE_Labels']==k]['ConvVAE_B'],c=col,alpha=0.0125)

ax0.scatter(Data[Data['ConvVAE_Labels']==-1]['ConvVAE_A'],Data[Data['ConvVAE_Labels']==-1]['ConvVAE_B'],c='black',alpha=0.0125)
ax0.set_title('Masha',loc='right')
PlotStyle(ax0)

k=0
for clust,colr in zip(clabels,localColors):
    
    axes = plt.subplot(gs[1,k])
    axes.hist(Data[Data['ConvVAE_Labels']==clust]['week'],bins=50,color='blue')
    PlotStyle(axes)
    k=k+1
    
plt.tight_layout()

nrows,ncolumns = GetGridShape(11)
subPlotIndexs=[(j,k) for j in range(nrows) for k in range(ncolumns)]

fig,axes=plt.subplots(nrows,ncolumns,figsize=(15,12),sharex=True)
for k,kal in enumerate(clabels):
    
    clusterData = Data[Data['ConvVAE_Labels']==kal]
    low = clusterData[clusterData['week']<0.5]['id']
    high = clusterData[clusterData['week']>0.5]['id']
    axes[subPlotIndexs[k]].hist(KmersData['A'].loc[high],color='red',bins=100,density=True,label='Cluster = '+str(kal))
    axes[subPlotIndexs[k]].hist(KmersData['A'].loc[low],color='red',bins=100,alpha=0.5,density=True,label='Cluster = '+str(kal))
    axes[subPlotIndexs[k]].set_xlim([0.05,0.15])
    axes[subPlotIndexs[k]].set_xlabel('Nucleotide content')
    axes[subPlotIndexs[k]].set_title('Cluster = ' + str(kal))
    PlotStyle(axes[subPlotIndexs[k]])
PlotStyle(axes[subPlotIndexs[-1]])
fig.suptitle('Adenine shift',x=0.9,y=0.9)

fig,axes=plt.subplots(nrows,ncolumns,figsize=(15,12),sharex=True)
for k,kal in enumerate(clabels):
    
    clusterData = Data[Data['ConvVAE_Labels']==kal]
    low = clusterData[clusterData['week']<0.5]['id']
    high = clusterData[clusterData['week']>0.5]['id']
    axes[subPlotIndexs[k]].hist(KmersData['C'].loc[high],color='blue',bins=100,density=True,label='Cluster = '+str(kal))
    axes[subPlotIndexs[k]].hist(KmersData['C'].loc[low],color='blue',bins=100,alpha=0.5,density=True,label='Cluster = '+str(kal))
    axes[subPlotIndexs[k]].set_xlim([0.2,0.35])
    axes[subPlotIndexs[k]].set_xlabel('Nucleotide content')
    axes[subPlotIndexs[k]].set_title('Cluster = ' + str(kal))
    PlotStyle(axes[subPlotIndexs[k]])
PlotStyle(axes[subPlotIndexs[-1]])
fig.suptitle('Cytosine shift',x=0.9,y=0.9)

fig,axes=plt.subplots(nrows,ncolumns,figsize=(15,12),sharex=True)
for k,kal in enumerate(clabels):
    
    clusterData = Data[Data['ConvVAE_Labels']==kal]
    low = clusterData[clusterData['week']<0.5]['id']
    high = clusterData[clusterData['week']>0.5]['id']
    axes[subPlotIndexs[k]].hist(KmersData['G'].loc[high],color='green',bins=100,density=True,label='Cluster = '+str(kal))
    axes[subPlotIndexs[k]].hist(KmersData['G'].loc[low],color='green',bins=100,alpha=0.5,density=True,label='Cluster = '+str(kal))
    axes[subPlotIndexs[k]].set_xlim([0.75,0.85])
    axes[subPlotIndexs[k]].set_xlabel('Nucleotide content')
    axes[subPlotIndexs[k]].set_title('Cluster = ' + str(kal))
    PlotStyle(axes[subPlotIndexs[k]])
PlotStyle(axes[subPlotIndexs[-1]])
fig.suptitle('Guanine shift',x=0.9,y=0.9)

fig,axes=plt.subplots(nrows,ncolumns,figsize=(15,12),sharex=True)
for k,kal in enumerate(clabels):
    
    clusterData = Data[Data['ConvVAE_Labels']==kal]
    low = clusterData[clusterData['week']<0.5]['id']
    high = clusterData[clusterData['week']>0.5]['id']
    axes[subPlotIndexs[k]].hist(KmersData['T'].loc[high],color='black',bins=100,density=True,label='Cluster = '+str(kal))
    axes[subPlotIndexs[k]].hist(KmersData['T'].loc[low],color='black',bins=100,alpha=0.5,density=True,label='Cluster = '+str(kal))
    axes[subPlotIndexs[k]].set_xlim([0.7,0.8])
    axes[subPlotIndexs[k]].set_xlabel('Nucleotide content')
    axes[subPlotIndexs[k]].set_title('Cluster = ' + str(kal))
    PlotStyle(axes[subPlotIndexs[k]])
PlotStyle(axes[subPlotIndexs[-1]])
fig.suptitle('Thymine/Uracil shift',x=0.91,y=0.9)

filled_markers = ('o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X')
maxAlt = Data[Data['SimplifiedGEO']=='USA']['geo_alt'].max()
minAlt = Data[Data['SimplifiedGEO']=='USA']['geo_alt'].min()
boundaries = np.linspace(minAlt,maxAlt,num=12)
colorCoding = np.linspace(0,1,num=11)

def GetElevationCoding(elevation):
    clr = 0
    for k in range(len(boundaries)-1):
        if elevation>boundaries[k] and elevation<boundaries[k+1]:
            clr = colorCoding[k]
            break
        
    return clr 

plt.figure(figsize=(15,10))   
for k,kal in enumerate(clabels):
    
    clusterData = Data[Data['ConvVAE_Labels']==kal]
    clusterData = clusterData[clusterData['week']<0.5]
    usData = clusterData[clusterData['SimplifiedGEO']=='USA']
    elevCoding = [GetElevationCoding(val) for val in usData['geo_alt']]
    plt.scatter(usData['geo_long'],usData['geo_lat'],marker=filled_markers[k],c=elevCoding)

plt.ylim([24,50])
plt.xlim([-130,-65])
plt.title('Geographical Encoding (First Half of the Year)',loc='right')
ImageStyle(plt.gca()) 

plt.figure(figsize=(15,10))   
for k,kal in enumerate(clabels):
    
    clusterData = Data[Data['ConvVAE_Labels']==kal]
    clusterData = clusterData[clusterData['week']>0.5]
    usData = clusterData[clusterData['SimplifiedGEO']=='USA']
    elevCoding = [GetElevationCoding(val) for val in usData['geo_alt']]
    plt.scatter(usData['geo_long'],usData['geo_lat'],marker=filled_markers[k],c=elevCoding)

plt.ylim([24,50])
plt.xlim([-130,-65])
plt.title('Geographical Encoding (Second Half of the Year)',loc='right')
ImageStyle(plt.gca()) 

###############################################################################
# Variant Encoding 
###############################################################################

plt.figure(figsize=(16,8))
plt.scatter(Data['PCA_A'],Data['PCA_B'],c=Data['simpPango_encoding'],alpha=0.05,cmap='viridis',label='Variant Encoding ')
plt.legend(loc=1,frameon=False,fontsize='13')
plt.title('Katya',loc='right')
PlotStyle(plt.gca()) 

plt.figure(figsize=(16,8))
plt.scatter(Data['VAE_A'],Data['VAE_B'],c=Data['simpPango_encoding'],alpha=0.05,cmap='viridis',label='Variant Encoding ')
plt.legend(loc=1,frameon=False,fontsize='13')
plt.title('Nina',loc='right')
PlotStyle(plt.gca()) 

plt.figure(figsize=(16,8))
plt.scatter(Data['ConvVAE_A'],Data['ConvVAE_B'],c=Data['simpPango_encoding'],alpha=0.05,cmap='viridis',label='Variant Encoding ')
plt.legend(loc=1,frameon=False,fontsize='13')
plt.title('Masha',loc='right')
PlotStyle(plt.gca()) 

plt.figure(figsize=(16,8))
plt.scatter(Data['PCA_A'],Data['PCA_B'],c=Data['binary_encoding'],alpha=0.05,cmap='viridis',label='Variant Encoding ')
plt.legend(loc=1,frameon=False,fontsize='13')
plt.title('Katya',loc='right')
PlotStyle(plt.gca()) 

plt.figure(figsize=(16,8))
plt.scatter(Data['VAE_A'],Data['VAE_B'],c=Data['binary_encoding'],alpha=0.05,cmap='viridis',label='Variant Encoding ')
plt.legend(loc=1,frameon=False,fontsize='13')
plt.title('Nina',loc='right')
PlotStyle(plt.gca()) 

plt.figure(figsize=(16,8))
plt.scatter(Data['ConvVAE_A'],Data['ConvVAE_B'],c=Data['binary_encoding'],alpha=0.15,cmap='viridis',label='Variant Encoding ')
plt.legend(loc=1,frameon=False,fontsize='13')
plt.title('Masha',loc='right')
PlotStyle(plt.gca()) 

###############################################################################
# Sampling bias
###############################################################################

plt.figure(figsize=(12,7))
plt.hist(Data['outbreaktime'],bins=75,density=True)
plt.xlabel('Time')
PlotStyle(plt.gca())

plt.figure(figsize=(12,7))
plt.hist(Data['week'],bins=50,density=True)
plt.xlabel('Week of the year')
PlotStyle(plt.gca())


plt.figure(figsize=(12,7))
plt.hist(Data['geo_lat'],bins=50,density=True)
plt.xlabel('Latitude')
PlotStyle(plt.gca())

plt.figure(figsize=(12,7))
plt.hist(Data['geo_long'],bins=50,density=True)
plt.xlabel('Longitude')
PlotStyle(plt.gca())

plt.figure(figsize=(12,7))
plt.hist(Data['geo_alt'],bins=50,density=True)
plt.xlabel('Altitude')
PlotStyle(plt.gca())

reSamplingIndex = GetEqualizedIndex(Data['outbreaktime'],bins=1000,maxCount=100)

plt.figure(figsize=(12,7))
plt.hist(Data['outbreaktime'].iloc[reSamplingIndex],bins=1000)
plt.xlabel('Time')
PlotStyle(plt.gca())

plt.figure(figsize=(16,8))
plt.scatter(Data['PCA_A'].iloc[reSamplingIndex],Data['PCA_B'].iloc[reSamplingIndex],c=Data['week'].iloc[reSamplingIndex],alpha=0.15,cmap='viridis',label='Time encoding')
plt.legend(loc=1,frameon=False,fontsize='13')
plt.title('Katya',loc='right')
PlotStyle(plt.gca()) 

plt.figure(figsize=(16,8))
plt.scatter(Data['VAE_A'].iloc[reSamplingIndex],Data['VAE_B'].iloc[reSamplingIndex],c=Data['week'].iloc[reSamplingIndex],alpha=0.15,cmap='viridis',label='Time encoding')
plt.legend(loc=1,frameon=False,fontsize='13')
plt.title('Nina',loc='right')
PlotStyle(plt.gca()) 

plt.figure(figsize=(16,8))
plt.scatter(Data['ConvVAE_A'].iloc[reSamplingIndex],Data['ConvVAE_B'].iloc[reSamplingIndex],c=Data['week'].iloc[reSamplingIndex],alpha=0.15,cmap='viridis',label='Time encoding')
plt.legend(loc=1,frameon=False,fontsize='13')
plt.title('Masha',loc='right')
PlotStyle(plt.gca()) 

plt.figure(figsize=(16,8))
plt.scatter(Data['PCA_A'].iloc[reSamplingIndex],Data['PCA_B'].iloc[reSamplingIndex],c=Data['binary_encoding'].iloc[reSamplingIndex],alpha=0.05,cmap='viridis',label='Variant Encoding ')
plt.legend(loc=1,frameon=False,fontsize='13')
plt.title('Katya',loc='right')
PlotStyle(plt.gca()) 

plt.figure(figsize=(16,8))
plt.scatter(Data['VAE_A'].iloc[reSamplingIndex],Data['VAE_B'].iloc[reSamplingIndex],c=Data['binary_encoding'].iloc[reSamplingIndex],alpha=0.05,cmap='viridis',label='Variant Encoding ')
plt.legend(loc=1,frameon=False,fontsize='13')
plt.title('Nina',loc='right')
PlotStyle(plt.gca()) 

plt.figure(figsize=(16,8))
plt.scatter(Data['ConvVAE_A'].iloc[reSamplingIndex],Data['ConvVAE_B'].iloc[reSamplingIndex],c=Data['binary_encoding'].iloc[reSamplingIndex],alpha=0.15,cmap='viridis',label='Variant Encoding ')
plt.legend(loc=1,frameon=False,fontsize='13')
plt.title('Masha',loc='right')
PlotStyle(plt.gca()) 

fig,axes=plt.subplots(nrows,ncolumns,figsize=(15,12),sharex=True)
for k,kal in enumerate(clabels):
    
    clusterData = Data.iloc[reSamplingIndex]
    clusterData = Data[Data['ConvVAE_Labels']==kal]
    low = clusterData[clusterData['week']<0.5]['id']
    high = clusterData[clusterData['week']>0.5]['id']
    axes[subPlotIndexs[k]].hist(KmersData['C'].loc[high],color='blue',bins=100,density=True,label='Cluster = '+str(kal))
    axes[subPlotIndexs[k]].hist(KmersData['C'].loc[low],color='blue',bins=100,alpha=0.5,density=True,label='Cluster = '+str(kal))
    axes[subPlotIndexs[k]].set_xlim([0.2,0.35])
    axes[subPlotIndexs[k]].set_xlabel('Nucleotide content')
    axes[subPlotIndexs[k]].set_title('Cluster = ' + str(kal))
    PlotStyle(axes[subPlotIndexs[k]])
PlotStyle(axes[subPlotIndexs[-1]])
fig.suptitle('Cytosine shift',x=0.9,y=0.9)

fig,axes=plt.subplots(nrows,ncolumns,figsize=(15,12),sharex=True)
for k,kal in enumerate(clabels):
    
    clusterData = Data.iloc[reSamplingIndex]
    clusterData = Data[Data['ConvVAE_Labels']==kal]
    low = clusterData[clusterData['week']<0.5]['id']
    high = clusterData[clusterData['week']>0.5]['id']
    axes[subPlotIndexs[k]].hist(KmersData['T'].loc[high],color='black',bins=100,density=True,label='Cluster = '+str(kal))
    axes[subPlotIndexs[k]].hist(KmersData['T'].loc[low],color='black',bins=100,alpha=0.5,density=True,label='Cluster = '+str(kal))
    axes[subPlotIndexs[k]].set_xlim([0.7,0.8])
    axes[subPlotIndexs[k]].set_xlabel('Nucleotide content')
    axes[subPlotIndexs[k]].set_title('Cluster = ' + str(kal))
    PlotStyle(axes[subPlotIndexs[k]])
PlotStyle(axes[subPlotIndexs[-1]])
fig.suptitle('Thymine/Uracil shift',x=0.91,y=0.9)
