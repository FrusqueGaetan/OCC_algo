"""
Usefull function :
    
generate_data -
Code for generating a simple dataset

Version: 30/04/2021
"""
import numpy as np
import pandas as pd
import tensorflow.keras as keras
import os
import sys


def generate_data(Nhealth=1200,Nabnormal=200):
    #Ntrain number of healthy data
    #Nabnormal number of abmormal data
    
    Ntime = 512
    
    Mhealth = 0.5*np.random.randn(Nhealth,Ntime)
    Mabnormal = 0.5*np.random.randn(Nabnormal,Ntime)
    
    #Sampling frequency of the model 5Khz
    #Normal signal, frequency between 500-750 Hz
    #Abnormal = frequency between 500-750 Hz + between 2000-2500 Hz
    
    Mhealth += np.sin(np.outer((np.random.rand(Nhealth)/10+0.2)*(np.pi),np.arange(0,Ntime,1)))
    Mabnormal += np.sin(np.outer((np.random.rand(Nabnormal)/10+0.2)*(np.pi),np.arange(0,Ntime,1))) + np.sin(np.outer((np.random.rand(Nabnormal)/5+0.8)*(np.pi),np.arange(0,Ntime,1)))
    
    return Mhealth, Mabnormal
 
def read_data_fold(folder):
    ld = os.listdir(folder)
    Table=[]
    X = []
    for file in ld:
        data = np.load(folder+file)
        df = pd.DataFrame(data)
        df.columns = ['time', 
                  'trigger', 
                  'pole_A', 
                  'pole_B', 
                  'pole_C', 
                  'coil_open', 
                  'coil_close', 
                  'acc_786A', 
                  'acc_ach01']
        Table.append(df)
        X.append(df['acc_786A'].values)
    return Table, X

    
def SimpleAE_model(inputDim):

    inputLayer = keras.layers.Input(shape=(inputDim,))
    h = keras.layers.Dense(64, activation="relu")(inputLayer)
    h = keras.layers.Dense(64, activation="relu")(h)
    h = keras.layers.Dense(8, activation="relu")(h)
    h = keras.layers.Dense(64, activation="relu")(h)
    h = keras.layers.Dense(64, activation="relu")(h)
    h = keras.layers.Dense(inputDim, activation=None)(h)

    return keras.models.Model(inputs=inputLayer, outputs=h)

def Desp2Spec(Coeff,Pool='Mean',NP=313):
    level=len(Coeff)
    R = np.zeros((level,NP))
    for j in range(level):
        X = Coeff[j]
        v = np.floor(np.linspace(0,np.shape(X)[0],NP+1))
        
        for k in range(NP):
            if Pool=='Mean':
                R[j,k] = 10 * np.log10(np.mean(X[int(v[k]):int((v[k+1])+1)]**2)+sys.float_info.epsilon)
            elif Pool=='Max':
                R[j,k] = 10 * np.log10(np.max(X[int(v[k]):int((v[k+1])+1)]**2)+sys.float_info.epsilon)
                
    return np.array(R)
        

