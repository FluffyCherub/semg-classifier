# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 02:55:00 2022

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

import csv
import numpy as np

global data_array
#global time_array
number_of_samples = 2000 #about 10-11 seconds
number_of_samples *= 50 # About 100 seconds
#data_array=['chonk']
data_array=[]
#time_array=[]

def update_stimulus(stimulus, t, start_1_time, stop_1_time, freq):
    start = start_1_time/freq
    stop = stop_1_time/freq
    #updates sepcified period
    for i in range(int(start), int(stop)):
        stimulus[i] = t
    return stimulus

def generate_stimulus(number_of_samples,time_taken):
    #generate initial stimulus set to 0 i.e. rest
    stimulus = [0]*number_of_samples
    freq = time_taken/number_of_samples
    
    #First gesture
    for i in range(1,11):
        time1 =10.0*i-5.0
        time2 =10.0*i
        stimulus = update_stimulus(stimulus, 1, time1, time2, freq)
    
    for i in range(11,21):
        time1 =10.0*i-5.0
        time2 =10.0*i
        stimulus = update_stimulus(stimulus, 2, time1, time2, freq)
        
    for i in range(21,31):
        time1 =10.0*i-5.0
        time2 =10.0*i
        stimulus = update_stimulus(stimulus, 3, time1, time2, freq)
    
    for i in range(31,41):
        time1 =10.0*i-5.0
        time2 =10.0*i
        stimulus = update_stimulus(stimulus, 4, time1, time2, freq)
        
    for i in range(41,51):
        time1 =10.0*i-5.0
        time2 =10.0*i
        stimulus = update_stimulus(stimulus, 5, time1, time2, freq)
    
    #stimulus = update_stimulus(stimulus, 1, 5.0, 10.0, freq)
    #stimulus = update_stimulus(stimulus, 1, 3.0, 4.0, freq)
    return stimulus

def check_if_process_running():
    try:
        for proc in psutil.process_iter():
            if proc.name()=='Myo Connect.exe':
                return True            
        return False
            
    except (psutil.NoSuchProcess,psutil.AccessDenied, psutil.ZombieProcess):
        print ("Myo Connect.exe", " not running")

# If the process Myo Connect.exe is not running then we restart that process
def restart_process():
    PROCNAME = "Myo Connect.exe"

    for proc in psutil.process_iter():
        # check whether the process name matches
        if proc.name() == PROCNAME:
            proc.kill()
            # Wait a second
            time.sleep(1)

    while(check_if_process_running()==False):
        path = r'C:\Program Files (x86)\Thalmic Labs\Myo Connect\Myo Connect.exe'
        os.startfile(path)
        time.sleep(1)

    print("Process started")
    return True

class Listener(myo.DeviceListener):
    def __init__(self, n):
        self.n = n
        self.lock = Lock()
        self.emg_data_queue = deque(maxlen=n)
        

    #Problem code
    def on_connected(self, event):
        print("Myo Connected")
        self.started = time.time()
        event.device.stream_emg(True)
        
    def get_emg_data(self):
        with self.lock:
            print("H")

    def on_emg(self, event):
        #event.device.stream_emg(True)
        with self.lock:
            self.emg_data_queue.append((event.emg))
            
            if len(list(self.emg_data_queue))>=number_of_samples:
                data_array.append(list(self.emg_data_queue))
                self.emg_data_queue.clear()
                return False
            
startTime = datetime.now()         

while True:
    try:
        myo.init()
        input("Make Gestures")
        #print("Make Gestures")
        hub = myo.Hub()
        listener = Listener(number_of_samples)
        #print("listener")
        startTime = datetime.now()  
        hub.run(listener.on_event, number_of_samples*10)
        #print(data_array)
        endTime = datetime.now()  
        emg_data = np.array((data_array[0]))
        #print("training_set")
        data_array.clear()
        #Get time data, use to make stimulus
        #Store time, emg and stimulus data into csv
        
        #print(emg_data[2])
        break
    except:
        print("Problem")   
        while(restart_process()!=True):
            pass
        # Wait for 3 seconds until Myo Connect.exe starts
        time.sleep(3)

time_taken = endTime - startTime

print("Time Taken: ",time_taken)
print("Sample rate:", number_of_samples/time_taken.total_seconds(), "per second")

stimulus = generate_stimulus(number_of_samples,time_taken.total_seconds())
stimulus = np.array(stimulus)

print("Stimulus:", stimulus.shape)

if emg_data.size:
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
    
    with open('unprocessed_emg_data_5.csv', 'w', newline='') as outcsv:
        unprocessed_emg_data = csv.writer(outcsv)
        unprocessed_emg_data.writerow(["Stimulus", "Emg1", "Emg2", "Emg3", "Emg4", "Emg5", "Emg6", "Emg7", "Emg8"])
    
        #unprocessed_emg_data.writerow(['John Smith', 'Accounting', 'November'])
        #unprocessed_emg_data.writerow(['Erica Meyers', 'IT', 'March'])
        for a in range((emg_data.shape[0])):
            unprocessed_emg_data.writerow([stimulus[a], emg_data[a,0], emg_data[a,1], emg_data[a,2], emg_data[a,3], emg_data[a,4], emg_data[a,5], emg_data[a,6], emg_data[a,7]])

    with open('unprocessed_emg_data_5.csv') as csv_file:
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
        


