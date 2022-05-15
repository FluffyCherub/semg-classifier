# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 06:41:33 2022

@author: user
"""
from collections import deque
from threading import Lock
from datetime import datetime

import matplotlib.pyplot as plt
import myo
import time
import psutil
import os
import sys

import csv
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns

stimulus = []
emg = []

# Opening data and storing in correct format
with open('unprocessed_emg_data_5_2.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            print(f'Column names are {", ".join(row)}')
            line_count += 1
        else:
            #print(f'\t{row[0]} works in the {row[1]} department, and was born in {row[2]}.')
            line_count += 1
            stimulus.append(int(row[0]))
            ro = [int(row[1]),int(row[2]),int(row[3]),int(row[4]),int(row[5]),int(row[6]),int(row[7]),int(row[8])]
            #print(r)
            emg.append(ro)
            
    print(f'Processed {line_count} lines.')
    
    stimulus = np.array(stimulus)
    emg = np.array(emg)
    print(stimulus.shape)
    print(emg.shape)
#print(emg)
#Process raw data
start = 0
end = round(len(stimulus),-1)
period = 200

rms_emg = np.empty((int(round(end/10)),0))
for b in range(8): #Number of blocks
    linend = 0
    rms = []
    #print("Hello!")
    for x in range(start, end, 10):
        linend = x + (period-10)
        if (linend > end):
            linend = end
        spread = emg[x:linend,b]
        #print(spread)
        s = spread**2
        m = np.mean(s)
        r = np.sqrt(m)
        rms.append(r)
    c = np.reshape(rms, (len(rms), 1))
    print("RMS shape:",rms_emg.shape,"RMS to add shape:", c.shape)
    rms_emg = np.append(rms_emg, c, axis=1)

#print(rms_emg)
print(rms_emg.shape)
print(stimulus.shape)

#Produces labels that match 10101 data set
labels = []
for y in range(4, len(stimulus), 10):
    s = stimulus[y]
    #print(s[0])
    #print(s)
    labels.append(s)
labels = np.array(labels)
print("labels new shape:", len(labels))

# fig, axs = plt.subplots(9)
# fig.suptitle('Stimulus vs rms_emg, all blocks,'+str(period)+' Hz smoothing')
# axs[0].plot(range(labels.shape[0]), labels , color ="orange")
# axs[1].plot(range(rms_emg.shape[0]), rms_emg[:,0] , color ="purple")
# axs[2].plot(range(rms_emg.shape[0]), rms_emg[:,1] , color ="orange")
# axs[3].plot(range(rms_emg.shape[0]), rms_emg[:,2] , color ="green")
# axs[4].plot(range(rms_emg.shape[0]), rms_emg[:,3] , color ="yellow")
# axs[5].plot(range(rms_emg.shape[0]), rms_emg[:,4] , color ="brown")
# axs[6].plot(range(rms_emg.shape[0]), rms_emg[:,5] , color ="purple")
# axs[7].plot(range(rms_emg.shape[0]), rms_emg[:,6] , color ="orange")
# axs[8].plot(range(rms_emg.shape[0]), rms_emg[:,7] , color ="green")
# plt.show() 

#Store processed data
with open('processed_emg_data.csv', 'w', newline='') as outcsv:
    processed_emg_data = csv.writer(outcsv)
    processed_emg_data.writerow(["Stimulus", "RMS1", "RMS2", "RMS3", "RMS4", "RMS5", "RMS6", "RMS7", "RMS8"])

    #unprocessed_emg_data.writerow(['John Smith', 'Accounting', 'November'])
    #unprocessed_emg_data.writerow(['Erica Meyers', 'IT', 'March'])
    for a in range((rms_emg.shape[0])):
        processed_emg_data.writerow([labels[a], rms_emg[a,0], rms_emg[a,1], rms_emg[a,2], rms_emg[a,3], rms_emg[a,4], rms_emg[a,5], rms_emg[a,6], rms_emg[a,7]])
with open('processed_emg_data.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            print(f'Column names are {", ".join(row)}')
            line_count += 1
        else:
            #print(f'\t{row[0]} works in the {row[1]} department, and was born in {row[2]}.')
            line_count += 1
    print(f'Processed {line_count} lines.')
#Split data
train_labels = []
test_labels = []
train_emg = []
test_emg = []
for z in range(len(rms_emg)):
    if z%2 == 0:
        #If even add to test set
        test_emg.append(rms_emg[z,:]) # Change : to pick specific features (0:4)
        test_labels.append(labels[z])
        
    else:
        #If odd add to train set
        train_emg.append(rms_emg[z,:]) # Change : to pick specific features (0:4)
        train_labels.append(labels[z])

train_emg = np.array(train_emg)
train_labels = np.array(train_labels)
test_emg = np.array(test_emg)
test_labels = np.array(test_labels)

#Check to see if shape matches for testing classifier
print("Train:", train_emg.shape,"Test: ",test_emg.shape)
print(train_labels.shape, test_labels.shape)

#Test data
#Test with KNN
model = KNeighborsClassifier(n_neighbors=2)
model.fit(train_emg,train_labels)
predicted_emg= model.predict(test_emg)
print(predicted_emg)
precent_accuracy = metrics.accuracy_score(test_labels, predicted_emg) *100
print("Accuracy (KNN):",precent_accuracy)

# cm = confusion_matrix(test_labels, predicted_emg)
# print (cm.shape)
# #cm = np.delete(cm, 0, 0)
# #cm = np.delete(cm, 0, 1)
# #a_list = list(range(1,cm.shape[0] +1))
# a_list = list(range(0,cm.shape[0]))
# ax = sns.heatmap(cm, annot=True, fmt='g', xticklabels=a_list, yticklabels=a_list);
# ax.set_title('Confusion Matrix - Nearest Neighbours');
# ax.set_xlabel('Predicted Movement')
# ax.set_ylabel('Actual Movement')
# print(a_list)
# plt.show()

#Test with MLP
#solver='lbfgs' best for small, switched to 'adam' cause its better for large
#65 (lri = 0.1) = 84.08, 75 = 82.62, 70 = 83.56, 67 =63.739999999999995
#lri = 0.01 => 90.94, lri = 0.005 => 93.62, lri (max iter increase to 500) = 0.001 => 92.60000000000001,lri (max iter to 400) = 0.003 => 91.46, lri (max iter to 500) = 0.003 => 91.46
#alpha=1e-4 => 93.62, alpha=1e-3 => 91.14, alpha=1e-5 => 91.24,
clf = MLPClassifier(solver='adam', alpha=1e-4, hidden_layer_sizes=(65,), random_state=1,learning_rate_init=0.005,max_iter=500)
clf.fit(train_emg, train_labels)
predicted_emg_mlp = clf.predict(test_emg)
precent_accuracy = metrics.accuracy_score(test_labels, predicted_emg_mlp) *100
print("Accuracy (MLP):",precent_accuracy)

cm = confusion_matrix(test_labels, predicted_emg_mlp)
print (cm.shape)
cm = np.delete(cm, 0, 0)
cm = np.delete(cm, 0, 1)
a_list = list(range(1,cm.shape[0] +1))
#a_list = list(range(0,cm.shape[0]))
ax = sns.heatmap(cm, annot=True, fmt='g', xticklabels=a_list, yticklabels=a_list);
ax.set_title('Confusion Matrix - Multilayer Perceptron');
ax.set_xlabel('Predicted Movement')
ax.set_ylabel('Actual Movement')
print(a_list)
plt.show()