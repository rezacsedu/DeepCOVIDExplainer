from __future__ import print_function
import numpy as np
import time
import csv
import os
import re

f=open("frontXRsideupMRIsideVGGXRsideVGGMRIupVGG-pos.csv","a")
readerMRI = open("frontXRsideupMRIsideVGGXRsideVGG-pos.csv")
dataMRI=readerMRI.readlines()
readerXR = open("VGG-MRIup-pos1.csv")
dataXR=readerXR.readlines()
for XR in dataXR:
    for MRI in dataMRI:   
        if MRI.split(',')[0]==XR.split(',')[0]:
            MRI=MRI.strip('\r\n')
            MRI=MRI.strip(',')
            #re.sub(r"(.*)+L,", ',',XR,1)
            #re.sub(r"(.*)+R,", ',',XR,1)
            print(XR.split(',')[0])
            f.write(MRI+','+XR)
            
