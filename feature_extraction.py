# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 18:53:32 2021

@author: user
"""

import numpy as np
from scipy.io import loadmat
import scipy.linalg
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns
from datetime import datetime

startTime = datetime.now()

#Load data
#annots = loadmat('data\s1\S1_A1_E1.mat')
#annots = loadmat('data\s1\S1_A1_E2.mat')
annots = loadmat('data\s1\S1_A1_E3.mat')

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
print(emg.shape) #numpy array
print(emg)
'''
plt.title("Stimulus graph")
plt.xlabel("X axis")
plt.ylabel("Y axis")
plt.plot(range(stimulus.shape[0]), stimulus , color ="green")
plt.show()'''
'''
plt.title("repetition graph")
plt.xlabel("X axis")
plt.ylabel("Y axis")
plt.plot(range(repetition.shape[0]), repetition , color ="yellow")
plt.show()


plt.title("Emg graph - first block's recording")
plt.xlabel("X axis")
plt.ylabel("Y axis")
plt.plot(range(emg.shape[0]), emg[:,0] , color ="purple")
plt.show()

plt.title("Emg graph - first block's recording - up to 10,000")
plt.xlabel("X axis")
plt.ylabel("Y axis")
plt.plot(range(10000), emg[:10000,0] , color ="purple")
plt.show()'''



start = 0
end = round(len(stimulus),-1)
#end = 101010 # Has got to be muliple of 10, got to be more than size of array

#First Action, 10 repetitions

# fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10) = plt.subplots(10)
# fig.suptitle('Emg subplots, all blocks')
# ax1.plot(range(emg[start:end,0].shape[0]), emg[start:end,0] , color ="purple")
# ax2.plot(range(emg[start:end,0].shape[0]), emg[start:end,1] , color ="orange")
# ax3.plot(range(emg[start:end,0].shape[0]), emg[start:end,2] , color ="green")
# ax4.plot(range(emg[start:end,0].shape[0]), emg[start:end,3] , color ="yellow")
# ax5.plot(range(emg[start:end,0].shape[0]), emg[start:end,4] , color ="brown")
# ax6.plot(range(emg[start:end,0].shape[0]), emg[start:end,5] , color ="purple")
# ax7.plot(range(emg[start:end,0].shape[0]), emg[start:end,6] , color ="orange")
# ax8.plot(range(emg[start:end,0].shape[0]), emg[start:end,7] , color ="green")
# ax9.plot(range(emg[start:end,0].shape[0]), emg[start:end,8] , color ="yellow")
# ax10.plot(range(emg[start:end,0].shape[0]), emg[start:end,9] , color ="brown")
# plt.show()

#rms each value, for all bands

rms_emg = np.empty((int(round(end/10)),0))
for b in range(10):
    linend = 0
    rms  = []
    for x in range(start, end, 10):
        linend = x + 90
        if (linend > end):
            linend = end
        #print(emg[x:linend,b])
        r = np.sqrt(np.mean(emg[x:linend,b]**2))
        rms.append(r)
    c = np.reshape(rms, (len(rms), 1))
    print("RMS shape:",rms_emg.shape,"RMS to add shape:", c.shape)
    rms_emg = np.append(rms_emg, c, axis=1)

#Produces labels that match 10101 data set
labels = []
new_repetition = []
for y in range(4, len(stimulus), 10):
    s = stimulus[y]
    t = repetition[y]
    #print(s[0])
    labels.append(s[0])
    new_repetition.append(t[0])
print("labels and new_repetition:", len(labels), len(new_repetition))

#First Action, 10 repetitions, first 5 bands, pre-processed, it works!!
'''
fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10) = plt.subplots(10)
fig.suptitle('Emg graph- first 5 bands - first action - preprocessed')
ax1.plot(range(rms_emg.shape[0]), rms_emg[:,0] , color ="purple")
ax2.plot(range(rms_emg.shape[0]), rms_emg[:,1] , color ="orange")
ax3.plot(range(rms_emg.shape[0]), rms_emg[:,2] , color ="green")
ax4.plot(range(rms_emg.shape[0]), rms_emg[:,3] , color ="yellow")
ax5.plot(range(rms_emg.shape[0]), rms_emg[:,4] , color ="brown")
ax6.plot(range(rms_emg.shape[0]), rms_emg[:,5] , color ="purple")
ax7.plot(range(rms_emg.shape[0]), rms_emg[:,6] , color ="orange")
ax8.plot(range(rms_emg.shape[0]), rms_emg[:,7] , color ="green")
ax9.plot(range(rms_emg.shape[0]), rms_emg[:,8] , color ="yellow")
ax10.plot(range(rms_emg.shape[0]), rms_emg[:,9] , color ="brown")
plt.show() 


plt.title("RMS graph - first block's recording - up to previous 10000")
plt.xlabel("X axis")
plt.ylabel("Y axis")
plt.plot(range(1000), rms_emg[:1000,0] , color ="purple")
plt.show()'''

# Data = rms_emg --> need to spilt into samples based on activity
# Labels = labels
# Repetitiom = whuch repetition it was lassidied as
# Split train and test data

print(rms_emg.shape)
print(len(new_repetition))

train_labels = []
test_labels = []
train_emg = []
test_emg = []
for z in range(len(new_repetition)-10):
    val = new_repetition[z]
    #print(val)
    
    if val == 2 or val == 5 or val == 7 or (val == 0 and z%2 == 0):
        test_emg.append(rms_emg[z,:]) # Change : to pick specific features (0:4)
        test_labels.append(labels[z])
        
    else:
        train_emg.append(rms_emg[z,:]) # Change : to pick specific features (0:4)
        train_labels.append(labels[z])
        
'''
    elif (val == 0 and z%10 != 0):
        rest_emg.append(rms_emg[z,:])'''

'''
plt.title("new_repetition graph")
plt.xlabel("X axis")
plt.ylabel("Y axis")
plt.plot(range(len(new_repetition)), new_repetition , color ="yellow")
plt.show()'''
        
train_emg = np.array(train_emg)
train_labels = np.array(train_labels)
test_emg = np.array(test_emg)
test_labels = np.array(test_labels)

#rest_emg = np.array(rest_emg)

#Check to see if shape matches for testing classifier
print("Train:", train_emg.shape,"Test: ",test_emg.shape)
print(train_labels.shape, test_labels.shape)

#KNN classifier 
'''
model = KNeighborsClassifier(n_neighbors=50)
model.fit(train_emg,train_labels)
predicted_emg= model.predict(test_emg)
print(predicted_emg)
precent_accuracy = metrics.accuracy_score(test_labels, predicted_emg) *100
print("Accuracy (KNN):",precent_accuracy)'''

#Mulitple layer perceptron classifier

clf = MLPClassifier(solver='lbfgs', alpha=1e-4, hidden_layer_sizes=(75,), random_state=1,learning_rate_init=0.1)
clf.fit(train_emg, train_labels)
predicted_emg_mlp = clf.predict(test_emg)
#print(predicted_emg_mlp[100])
precent_accuracy = metrics.accuracy_score(test_labels, predicted_emg_mlp) *100
print("Accuracy (MLP):",precent_accuracy)

#cm = confusion_matrix(test_labels, predicted_emg)
#print (cm.shape)
#cm_new = np.delete(cm, 0, 0)
#cm_new = np.delete(cm_new, 0, 1)
#a_list = list(range(1,cm_new.shape[0] +1))
#ax = sns.heatmap(cm_new, annot=True, fmt='g', xticklabels=a_list, yticklabels=a_list);
#ax.set_title('Confusion Matrix - Nearest Neighbours');
#ax.set_xlabel('Predicted Movement')
#ax.set_ylabel('Actual Movement')
#print(a_list)
#plt.show()


#cm = confusion_matrix(test_labels, predicted_emg_mlp)
#print (cm.shape)
#cm_new = np.delete(cm, 0, 0)
#cm_new = np.delete(cm_new, 0, 1)
#a_list = list(range(1,cm_new.shape[0] +1))
#ax = sns.heatmap(cm_new, annot=True, fmt='g', xticklabels=a_list, yticklabels=a_list);
#ax.set_title('Confusion Matrix - Mulitple Layer Perceptron');
#ax.set_xlabel('Predicted Movement')
#ax.set_ylabel('Actual Movement')
##print(a_list)
#plt.show()

print("Time Taken: ",datetime.now() - startTime)