#%%Import


import os

PathOCCToolbox = "C:/Users/gfrusque/Spyder_Part1/OCC"
os.chdir(PathOCCToolbox )
sys.path.insert(1, PathOCCToolbox+"/lib")

import numpy as np
import pandas as pd

# Graphics
import matplotlib as mplt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

import scipy.signal as scisi
import neurokit2

import HELM
from Utils import generate_data, SimpleAE_model

import tensorflow as tf

import librosa
import librosa.core
import librosa.feature
import sys

#%%

### Step A : set the training and validation dataset

##1 Consider a Healthy dataset and an unealthy one (for validation only)
nHealth=20000
nAbnormal=200
DataHealth, DataAbnormal = generate_data(Nhealth=nHealth,Nabnormal=nAbnormal)

#Build training dataset constitued only from Healthy data

##2
#2.1 Select randomly a subpart of the Healthy dataset
nHealth = np.shape(DataHealth)[0]
idTrainH = np.random.choice(np.arange(nHealth),size=(int(0.99*nHealth)),replace=False)

#2.2Create the Training dataset
DataTrain = DataHealth[idTrainH,:]

##3 Create the validation dataset, and validation label by mixing the 
#rest of the Heatly data + abnormal data
idTestH  = np.setdiff1d(np.arange(nHealth),idTrainH)
DataValid = np.concatenate((DataHealth[idTestH,:],DataAbnormal),axis=0)

LabelValid = np.concatenate((np.zeros((np.shape(idTestH)[0])),np.ones((nAbnormal,))))


#%%



### Step B : Feature selection

#Simple fourier transform
Ftrain = np.abs(np.fft.fft(DataTrain,axis=1))[:,0:256]
Fvalid = np.abs(np.fft.fft(DataValid,axis=1))[:,0:256]

#%%

### Step C : Training


#HELM Model: Michau, Gabriel, Yang Hu, Thomas Palmé, and Olga Fink. “Feature Learning for Fault Detection in High-Dimensional Condition-Monitoring Signals.” ArXiv Preprint ArXiv:1810.05550, 2018.
paraHELM={}
paraHELM['nhelm']          = 40                  # number of times HELM is trained and ran
paraHELM['fista_weight']   = 1e-3               # weight factor lambda for l1 norm reg (AE)
paraHELM['fista_cv']       = 1e-5               # Number of iterations or RMSE @ cv
paraHELM['ridge_weight']   = 1e-3              # weight factor lambda for l2 norm reg (1-class)
quant = 99.5
paraHELM['neuron_number']  = np.array([50])
elmT   = HELM.HELM(paraHELM, Ftrain)

#Simple Autoencoder with decision on the residual

SAEmodel = SimpleAE_model(np.shape(Ftrain)[1])
SAEmodel.compile(optimizer='adam',loss='mean_squared_error')
history = SAEmodel.fit(
    Ftrain,
    Ftrain,
    epochs=100,
    batch_size=512,
    shuffle=True,
    validation_split=0.1,
    verbose=1)
#plt.plot(history.history["loss"])


#%%

### Step D : Validation
outSAE = np.mean(np.square(Fvalid - SAEmodel.predict(Fvalid)), axis=1)
ScoreAUC_SAE = metrics.roc_auc_score(LabelValid, outSAE)
print(ScoreAUC_SAE)
#HELM Model
outHELM = HELM.HELM_run(elmT, Test = Fvalid)
res = outHELM['Test']['Y']
ScoreAUC_HELM = metrics.roc_auc_score(LabelValid, res)
print(ScoreAUC_HELM)

