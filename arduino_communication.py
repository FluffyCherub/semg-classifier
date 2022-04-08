# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 22:41:36 2022

@author: user
"""
import serial
import time

arduino = serial.Serial(port='COM4', baudrate=115200, timeout=.1)


def write_read(x):
    arduino.write(bytes(x, 'utf-8'))
    time.sleep(0.05)
    data = arduino.readline()
    return data


while True:
    num = input("Enter a number: ")
    value = write_read(num)
    print(value)