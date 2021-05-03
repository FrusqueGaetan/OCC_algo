"""
Usefull function :
    
generate_data -
Code for generating a simple dataset

Version: 30/04/2021
"""
import numpy as np
import tensorflow.keras as keras


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
    
def SimpleAE_model(inputDim):

    inputLayer = keras.layers.Input(shape=(inputDim,))
    h = keras.layers.Dense(64, activation="relu")(inputLayer)
    h = keras.layers.Dense(64, activation="relu")(h)
    h = keras.layers.Dense(8, activation="relu")(h)
    h = keras.layers.Dense(64, activation="relu")(h)
    h = keras.layers.Dense(64, activation="relu")(h)
    h = keras.layers.Dense(inputDim, activation=None)(h)

    return keras.models.Model(inputs=inputLayer, outputs=h)
