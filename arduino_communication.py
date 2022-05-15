# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 22:41:36 2022
# For lighting up pins and communicating with robots test
@author: user
"""
import serial
import time

# arduino = serial.Serial(port='COM4', baudrate=115200, timeout=.1) # LEDs
arduino = serial.Serial(port='COM5', baudrate=115200, timeout=.1) # Robot


def write_read(x):
    arduino.write(bytes(x, 'utf-8'))
    time.sleep(0.05)
    data = arduino.readline()
    return data


while True:
    num = input("Enter an integer: ")
    if num.isdigit():
        value = write_read(num)
        print(value)
    elif num.lower() == "end":
        print("End.")
        break
    else:
        print("Not an integer.")