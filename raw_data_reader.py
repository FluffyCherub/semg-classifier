# -*- coding: utf-8 -*-
"""
Created on Sun May  8 13:19:09 2022

@author: user
"""
import csv
import numpy as np
import matplotlib.pyplot as plt

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
    emg_data = np.array(emg)
    print(stimulus.shape)
    print(emg_data.shape)
    fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9) = plt.subplots(9)
    fig.suptitle('stimulus vs emg, all blocks')
    ax1.plot(range(stimulus.shape[0]), stimulus , color ="orange")
    ax2.plot(range(emg_data.shape[0]), emg_data[:,0] , color ="purple")
    ax3.plot(range(emg_data.shape[0]), emg_data[:,1] , color ="purple")
    ax4.plot(range(emg_data.shape[0]), emg_data[:,2] , color ="purple")
    ax5.plot(range(emg_data.shape[0]), emg_data[:,3] , color ="purple")
    ax6.plot(range(emg_data.shape[0]), emg_data[:,4] , color ="purple")
    ax7.plot(range(emg_data.shape[0]), emg_data[:,5] , color ="purple")
    ax8.plot(range(emg_data.shape[0]), emg_data[:,6] , color ="purple")
    ax9.plot(range(emg_data.shape[0]), emg_data[:,7] , color ="purple")
    plt.show()