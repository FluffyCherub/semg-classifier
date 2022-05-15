# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 20:59:02 2022

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
#from sklearn import metrics
#from sklearn.neural_network import MLPClassifier
#from sklearn.metrics import confusion_matrix
import serial
#import time

arduino = serial.Serial(port='COM4', baudrate=115200, timeout=.1) # LEDs
# arduino = serial.Serial(port='COM5', baudrate=115200, timeout=.1) # Robot

global data_array
global number_of_samples
global time_array

time_array = []
data_array=[]
number_of_samples = 50 # 20 ~0.5s

def write_read(x):
    arduino.write(bytes(x, 'utf-8'))
    time.sleep(0.05)
    data = arduino.readline()
    return data

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

    print("Process started")
    return True

class Listener(myo.DeviceListener):
    def __init__(self, n):
        self.n = n
        self.lock = Lock()
        self.emg_data_queue = deque(maxlen=n)
        self.time_data_queue = deque(maxlen=n)
        
    def on_connected(self, event):
        #print("Myo Connected")
        self.started = time.time()
        event.device.stream_emg(True)
        
    def get_emg_data(self):
        with self.lock:
            print("H")

    def on_emg(self, event):
        #event.device.stream_emg(True)
        with self.lock:
            self.emg_data_queue.append((event.emg))
            self.time_data_queue.append((datetime.now()))
            
            
            if len(list(self.emg_data_queue))>=number_of_samples:
                data_array.append(list(self.emg_data_queue))
                time_array.append(list(self.time_data_queue))
                self.emg_data_queue.clear()
                self.time_data_queue.clear()
                return False
labels = []
rms_emg = []    
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
            labels.append(int(row[0]))
            ro = [float(row[1]),float(row[2]),float(row[3]),float(row[4]),float(row[5]),float(row[6]),float(row[7]),float(row[8])]
            #print(r)
            rms_emg.append(ro)
    print(f'Processed {line_count} lines.')

train_rms = np.array(rms_emg)
train_labels = np.array(labels)
model = KNeighborsClassifier(n_neighbors=5)
model.fit(train_rms,train_labels)

#predicted= model.predict(test_rms)

def get_predicted():
    while True:
        try:
            myo.init()
            print("Gesture")
            hub = myo.Hub()
            listener = Listener(number_of_samples)
            startTime = datetime.now()  
            hub.run(listener.on_event, number_of_samples*10)
            endTime = datetime.now()  
            emg = np.array((data_array[0]))
            times = np.array((time_array[0]))
            data_array.clear()
            time_array.clear()
            #Process data as before
            start = 0
            end = round(len(emg),-1)
            rms_emg = np.empty((int(round(end/10)),0))
            period = 200
            for b in range(8): #Number of blocks
                linend = 0
                rms = []
                for x in range(start, end, 10):
                    linend = x + (period-10)
                    if (linend > end):
                        linend = end
                    r = np.sqrt(np.mean(emg[x:linend,b]**2))
                    rms.append(r)
                c = np.reshape(rms, (len(rms), 1))
                #print("RMS shape:",rms_emg.shape,"RMS to add shape:", c.shape)
                rms_emg = np.append(rms_emg, c, axis=1)
            predicted= model.predict(rms_emg)
            
            #Fix accompoying times
            temp = []
            for y in range(4, len(times), 10):
                s = times[y]
                temp.append(s)
            times = np.array(temp)
            
            #print(predicted)
            return predicted, times
            break
        except:
            print("Problem")   
            while(restart_process()!=True):
                pass
            # Wait for 3 seconds until Myo Connect.exe starts
            time.sleep(3)

#predicted = np.array([])
#times = np.array([])
predicted, times = get_predicted()
#print(predicted, times)

# plt.axis([0, 10, 0, 1])

timeout = 1.0 #seconds
try:
    while True:
        timeout_start = time.time()
        
        p, t = get_predicted()
        predicted = np.append(predicted, p)
        times = np.append(times, t)
        # if time.time() > (timeout_start + timeout):
        #     now = datetime.now()
        #     date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
        #     print("Ran out of time at: ", date_time)
        #     pass
        
        #Communicate with arudino
        c = 0
        total = 0
        for num in p:
            c+=1
            total += num 
        value = write_read(str(int(round(total/c))))
        print(value, times[-1])
        
        for num in p:
            value = write_read(str(num))
        #Plotting values
        #print(p, t)
        #plt.scatter(times, predicted, c='red', marker='x')
        #plt.plot(times, predicted)
        #plt.pause(0.01)
except KeyboardInterrupt:
    print('interrupted!')
    try:
        sys.exit(0)
    except SystemExit:
        os._exit(0)
#plt.show()