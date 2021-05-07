#Variational Mode decomposition
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 19:24:58 2019
@author: Vinícius Rezende Carvalho
"""
import numpy as np  
import matplotlib.pyplot as plt  
import os 
import sys

PathOCCToolbox = "C:/Users/gfrusque/Spyder_Part1/OCC/OCC_algo"
os.chdir(PathOCCToolbox )
sys.path.insert(1, PathOCCToolbox+"/lib")
import VMD  
from Utils import read_data_fold


#%%
#. Time Domain 0 to T  
T = 1000  
fs = 1/T  
t = np.arange(1,T+1)/T  
freqs = 2*np.pi*(t-0.5-fs)/(fs)  

#. center frequencies of components  
f_1 = 2  
f_2 = 24  
f_3 = 288  

#. modes  
v_1 = (np.cos(2*np.pi*f_1*t))  
v_2 = 1/4*(np.cos(2*np.pi*f_2*t))  
v_3 = 1/16*(np.cos(2*np.pi*f_3*t))  

f = v_1 + v_2 + v_3 + 0.1*np.random.randn(v_1.size)  

#. some sample parameters for VMD  
alpha = 2000       # moderate bandwidth constraint   <-- if alpha increase enforce the bandwith of the IMF to be smaller
tau = 0.            # noise-tolerance (no strict fidelity enforcement)  
K = 3              # 3 modes  
DC = 0             # no DC part imposed  (DC = 0 freq or mean of the signal )
init = 1           # initialize omegas uniformly  
tol = 1e-7  


#. Run VMD 
u, u_hat, omega = VMD.VMD(f, alpha, tau, K, DC, init, tol)  

#. Visualize decomposed modes
plt.figure()
plt.subplot(2,2,1)
plt.plot(f)
plt.title('Original signal')
plt.xlabel('time (s)')
plt.subplot(2,2,2)
plt.plot(u.T)
plt.title('Decomposed modes')
plt.xlabel('time (s)')
plt.legend(['Mode %d'%m_i for m_i in range(u.shape[0])])
plt.subplot(2,2,3)
plt.plot(np.abs(np.fft.fft(f))[0:int(T/2)])
plt.title('Spectrum')
plt.xlabel('Hz')
plt.subplot(2,2,4)
plt.plot(np.abs(np.fft.fft(u.T,axis=0))[0:int(T/2)])
plt.title('Mode Spectrum')
plt.xlabel('Hz')
plt.legend(['Mode %d'%m_i for m_i in range(u.shape[0])])
plt.tight_layout()




#%% Application on your data





#VMD 
alpha = 10000      # moderate bandwidth constraint  <-- if alpha increase enforce the bandwith of the IMF to be smaller
tau = 0.            # noise-tolerance (no strict fidelity enforcement)  
K = 8         # 6 modes  
DC = 0             # no DC part imposed  (DC = 0 freq or mean of the signal )
init = 1           # initialize omegas uniformly  
tol = 1e-7  


Table, X = read_data_fold("C:/Users/gfrusque/Spyder_Part1/OCC/OCC_algo/Data_Circuit_Breaker_04052021/")


IMF=[]
for i in range(1):#np.shape(X)[0]
    u, u_hat, omega = VMD.VMD(X[i], alpha, tau, K, DC, init, tol)  
    IMF.append(u)


#. Visualize result for one signal 
f = X[0]
u = IMF[0]
T = np.shape(f)[0]


plt.figure()
plt.subplot(2,2,1)
plt.plot(f)
plt.title('Original signal')
plt.xlabel('time (s)')
plt.subplot(2,2,2)
plt.plot(u.T)
plt.title('Decomposed modes')
plt.xlabel('time (s)')
plt.legend(['Mode %d'%m_i for m_i in range(u.shape[0])])
plt.subplot(2,2,3)
plt.plot(np.abs(np.fft.fft(f))[0:int(T/2)])
plt.title('Spectrum')
plt.xlabel('Hz')
plt.subplot(2,2,4)
plt.plot(np.abs(np.fft.fft(u.T,axis=0))[0:int(T/2)])
plt.title('Mode Spectrum')
plt.xlabel('Hz')
plt.legend(['Mode %d'%m_i for m_i in range(u.shape[0])])
plt.tight_layout()
