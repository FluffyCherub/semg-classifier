# -*- coding: utf-8 -*-
"""
Created on Sun Oct 31 00:45:52 2021

@author: Laney
"""
import numpy as np
from scipy.io import loadmat
import scipy.linalg
import matplotlib.pyplot as plt

#Load data
annots = loadmat('data\s1\S1_A1_E1.mat')

# Get data
header = annots['__header__']
version = annots['__version__']
globals_ = annots['__globals__']
emg = annots['emg']
stimulus = annots['stimulus']
glove = annots['glove']
subject = annots['subject']
exercise = annots['exercise']
repetition = annots['repetition']
restimulus = annots['restimulus']
rerepetition = annots['rerepetition']

#Checking data content
print(emg) #emg important
print(emg[15]) 
#print(stimulus[100000:101000])
print(emg.shape)#



plt.title("Stimulus graph")
plt.xlabel("X axis")
plt.ylabel("Y axis")
plt.plot(range(stimulus.shape[0]), stimulus , color ="blue")
plt.show()

plt.title("Emg graph")
plt.xlabel("X axis")
plt.ylabel("Y axis")
plt.plot(range(emg.shape[0]), emg[:,0] , color ="purple")
plt.show()

#First Action, 10 repetitions
fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5)
fig.suptitle('Emg subplots, first sub-action')
end = 10000
start = 0
ax1.plot(range(emg[start:end,0].shape[0]), emg[start:end,0] , color ="purple")
ax2.plot(range(emg[start:end,0].shape[0]), emg[start:end,1] , color ="orange")
ax3.plot(range(emg[start:end,0].shape[0]), emg[start:end,2] , color ="green")
ax4.plot(range(emg[start:end,0].shape[0]), emg[start:end,3] , color ="yellow")
ax5.plot(range(emg[start:end,0].shape[0]), emg[start:end,4] , color ="brown")

#Second Action, 10 repetitions
fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5)
fig.suptitle('Emg subplots, second action')
end = 20000
start = 10000 
ax1.plot(range(emg[start:end,0].shape[0]), emg[start:end,0] , color ="purple")
ax2.plot(range(emg[start:end,0].shape[0]), emg[start:end,1] , color ="orange")
ax3.plot(range(emg[start:end,0].shape[0]), emg[start:end,2] , color ="green")
ax4.plot(range(emg[start:end,0].shape[0]), emg[start:end,3] , color ="yellow")
ax5.plot(range(emg[start:end,0].shape[0]), emg[start:end,4] , color ="brown")


