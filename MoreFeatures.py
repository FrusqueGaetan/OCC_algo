import os
import sys

PathOCCToolbox = "C:/Users/gfrusque/Spyder_Part1/OCC/OCC_algo"
os.chdir(PathOCCToolbox )
sys.path.insert(1, PathOCCToolbox+"/lib")

import numpy as np
import pandas as pd

# Graphics
import matplotlib as mplt
import matplotlib.pyplot as plt
from sklearn import metrics

from sklearn import metrics
import HELM
from Utils import generate_data, SimpleAE_model, Desp2Spec

import tensorflow as tf

import librosa
import librosa.core
import librosa.feature
import librosa.display

import pywt


#%%
from scipy.signal import chirp

t = np.linspace(0, 10, 100000)
w = chirp(t, f0=5000, f1=0.1, t1=10, method='linear')
plt.plot(t, w)
plt.title("Linear Chirp, f(0)=6, f(10)=1")
plt.xlabel('t (sec)')
plt.show()



#Stft
D = librosa.stft(y=w,
                 n_fft=500,
                 hop_length=500)
D = 10 * np.log10(np.abs(D)**2+sys.float_info.epsilon)
librosa.display.specshow(D, sr=1000, hop_length=500,  x_axis='time', y_axis='linear')

#Mel spectrogram
D = librosa.feature.melspectrogram(y=w,
                               sr=10000,
                               n_fft=500,
                               hop_length=500,
                               n_mels=64,
                               power=2.0)
D = 10 * np.log10(D+sys.float_info.epsilon)
librosa.display.specshow(D, sr=10000,   x_axis='time', y_axis='mel')

#Melcesptrum
D = librosa.feature.mfcc(y=w, 
                     sr=10000 , 
                     hop_length=500, 
                     n_fft=500, 
                     n_mfcc=13, 
                     dct_type=2)
librosa.display.specshow(D, x_axis='time')

#Chromagram
D = librosa.feature.chroma_stft(y=w, 
                            sr=10000, 
                            hop_length=500, 
                            n_fft=500,
                            n_chroma=12)
librosa.display.specshow(D, y_axis='chroma', x_axis='time')


#Other features (see librosa website)
plt.figure(1,figsize=(15, 10))
n_fft = 500
hop_length = 500
C=librosa.feature.rms(y=w, frame_length=n_fft, hop_length=hop_length)
plt.subplot(3,3,1)
plt.plot(C.transpose())
plt.title('RMS')
C=librosa.feature.spectral_centroid(y=w, n_fft=n_fft, hop_length=hop_length)
plt.subplot(3,3,2)
plt.plot(C.transpose())
plt.title('Spectral centroid')
C=librosa.feature.spectral_bandwidth(y=w, n_fft=n_fft, hop_length=hop_length)
plt.subplot(3,3,3)
plt.plot(C.transpose())
plt.title('Spectral bandwith')
#    C[3,:]=librosa.feature.spectral_contrast(y=y, n_fft=n_fft, hop_length=hop_length)
C=librosa.feature.spectral_flatness(y=w, n_fft=n_fft, hop_length=hop_length)
plt.subplot(3,3,4)
plt.plot(C.transpose())
plt.title('Spectral flatness')
C=librosa.feature.spectral_rolloff(y=w, n_fft=n_fft, hop_length=hop_length)
plt.subplot(3,3,5)
plt.plot(C.transpose())
plt.title('Spectral roloff')
C=librosa.feature.zero_crossing_rate(y=w,  frame_length=n_fft, hop_length=hop_length)
plt.subplot(3,3,6)
plt.plot(C.transpose())
plt.title('Zeros crossing rate')



#%%Wavelet transform

filterd = 'db8'

Lvl=6
Coeffs = pywt.WaveletPacket(w, filterd, mode='symmetric')
D = np.array([node.data for node in Coeffs.get_level(Lvl, 'freq')])
D = Desp2Spec(D,NP=200)
librosa.display.specshow(D, sr=128, hop_length=500,  x_axis='time', y_axis='linear')



Coeffs = pywt.wavedec(w, filterd, mode='symmetric', level=11, axis=-1)
D = Desp2Spec(Coeffs,Pool='Mean',NP=200)
librosa.display.specshow(D, sr=22, hop_length=500,  x_axis='time', y_axis='linear')

 


# =============================================================================
# widths = np.arange(1, 31)
# coeffs, freqs = pywt.cwt(w, widths, 'mexh')
# plt.matshow(coef)
# plt.show()
# =============================================================================


