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
# loading packages
###############################################################################

import time
import requests
import pandas as pd

from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

###############################################################################
# Merging Data 
###############################################################################

GlobalDirectory=r"/media/tavoglc/storage/storage/datasets/DABE/LifeSciences/COVIDSeqs/nov2021"

df01 = pd.read_csv(GlobalDirectory+'/LatentRepresentations/PCA.csv')
df02 = pd.read_csv(GlobalDirectory+'/LatentRepresentations/VAELatent.csv')
df03 = pd.read_csv(GlobalDirectory+'/LatentRepresentations/ConvVAELatent.csv')

df01['id'] = [val[0:-2] for val in df01['id']]
df02['id'] = [val[0:-2] for val in df02['id']]
df03['id'] = [val[0:-2] for val in df03['id']]

df01 = df01.set_index('id')
df02 = df02.set_index('id')
df03 = df03.set_index('id')

df02.rename({'Latent0':'VAE_A','Latent1':'VAE_B'},axis=1,inplace=True)
df03.rename({'Latent0':'ConvVAE_A','Latent1':'ConvVAE_B'},axis=1,inplace=True)

collection = pd.concat([df01, df02,df03], axis=1, join="inner")
collIndex = [val for val in collection.index]

metaDataDF = pd.read_csv(GlobalDirectory+'/secuencias/sequences.csv')
metaDataDF.rename({'Accession':'id'},axis=1,inplace=True)
metaDataDF = metaDataDF.set_index('id')

selectedDF = metaDataDF.loc[collIndex]

selectedDF['Collection_Date'] = pd.to_datetime(selectedDF['Collection_Date'], format="%Y-%m-%d").dt.tz_localize(None)
selectedDF['Release_Date'] = pd.to_datetime(selectedDF['Release_Date'], format="%Y-%m-%d").dt.tz_localize(None)

###############################################################################
# Time Features
###############################################################################

initialOutbreak = selectedDF.loc['NC_045512']['Release_Date']
selectedDF['outbreaktime'] = [((val-initialOutbreak).days)/671 for val in selectedDF['Collection_Date']]
selectedDF['month'] = selectedDF['Collection_Date'].dt.month/12
selectedDF['week'] = selectedDF['Collection_Date'].dt.isocalendar().week/53

###############################################################################
# Geo location Features
###############################################################################

def GetSimplifiedCountry(GeoLocation):
    
    GeoLocation = str(GeoLocation)
    location = GeoLocation.find(':')
    if location ==-1:
        return GeoLocation
    else:
        return GeoLocation[0:location]
    
def GetElevation(lat, long):
    #modified from stackoverflow https://stackoverflow.com/questions/19513212/can-i-get-the-altitude-with-geopy-in-python-with-longitude-latitude
    query = ('https://api.open-elevation.com/api/v1/lookup'
             f'?locations={lat},{long}')
    r = requests.get(query).json()  # json object, various ways you can extract value
    # one approach is to use pandas json functionality:
    elevation = pd.json_normalize(r, 'results')['elevation'].values[0]
    return elevation

selectedDF['SimplifiedGEO'] = [GetSimplifiedCountry(val) for val in selectedDF['Geo_Location']]

###############################################################################
# Time Features
###############################################################################

locations = [str(val) for val in selectedDF['Geo_Location']]
uniqueLocations = list(set(locations)) 

geoCoords = []
geolocator = Nominatim(user_agent="locations")
geocode = RateLimiter(geolocator.geocode,min_delay_seconds=2)
                         
for val in uniqueLocations:
    
    location = geocode(val) 
    if type(location)==type(None):
        
        geoCoords.append((0,0,0))
        time.sleep(2.5)
        
    else:
        
        lat = location.latitude
        long = location.longitude
        alt = GetElevation(lat,long)
        
        geoCoords.append((lat,long,alt))
        time.sleep(2.5)
    
nameToGeo = dict([(val,sal) for val,sal in zip(uniqueLocations,geoCoords)])

selectedDF['geo_lat'] = [nameToGeo[str(val)][0] for val in selectedDF['Geo_Location']]
selectedDF['geo_long'] = [nameToGeo[str(val)][1] for val in selectedDF['Geo_Location']]
selectedDF['geo_alt'] = [nameToGeo[str(val)][2] for val in selectedDF['Geo_Location']]

headers = ['Submitters','Geo_Location','outbreaktime','month','week','SimplifiedGEO','Pangolin','geo_lat','geo_long','geo_alt']
mergedDF = pd.concat([collection,selectedDF[headers]], axis=1, join="inner")
mergedDF.to_csv(GlobalDirectory+'/mergedDF.csv')
