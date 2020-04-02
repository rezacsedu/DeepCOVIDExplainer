from __future__ import print_function
import numpy as np
import time
import csv
import os
from sklearn.metrics import classification_report
import sklearn.metrics as sklm
import math
from scipy import stats
  
def dense_to_one_hot(labels_dense,num_clases=5):
  return np.eye(num_clases)[labels_dense]

def load_val(): #average prediction class
  labelList=[]
  reader = open("/data/jiao/singlefront.csv") #prediction file path
  data=reader.readlines()
  label="q"
  for row in data:
     label=max(int(row.split(",")[0]))
     #label=math.ceil((int(row.split(",")[0])+int(row.split(",")[2]))/2)
     labelList.append(label)
  return np.array(labelList)

def load_v(): #prediction maximum
  labelList=[]
  imlist=[]
  reader = open("/data/jiao/DenseNet-cls.csv")#prediction file path
  data=reader.readlines()
  label="q"
  for row in data:
     for i in range(4):
        imlist.append(int(row.split(",")[i]))
     label=np.bincount(imlist)
     imlist=[]
     labelList.append(np.argmax(label))
  return np.array(labelList)


def load_va():#average softmax possibilities
  imlist=[]
  labelList=[]
  reader = open("/data/jiao/DenseNet-posfront.csv")#prediction file path
  data=reader.readlines()
  label="q"
  for row in data:
     for i in range(4):
        imlist.append(float(row.split(",")[i+12]))
     label=imlist.index(max(imlist))
     imlist=[]
     labelList.append(label)
  return np.array(labelList)

def load_valY():
  imgList=[]
  labelList=[]
  reader = open("/data/jiao/newlabel.csv") #original label file path
  data=reader.readlines()
  files = os.listdir('/data/jiao/XR/front/validation/')#test set image path
  for file in files:
        if file.endswith(".xml"):continue
        patient=file.split('_')[0]
        direction=file.split('_')[1].split('.')[0]
        label="q"
        for row in data:
           if patient in row.split(",")[0]:
              if "L" in direction:
                 label=row.split(",")[3]
              else:
                 label=row.split(",")[6]
              break
        if "V" in file:
                       label="3"
        if "8" not in label and "9" not in label and "X" not in label and '.' not in label:
          #if "." in label:
            #label='4'
          labelList.append(int(label))
  return np.array(labelList)

print(sklm.accuracy_score(load_valY(),load_va()))
print(sklm.classification_report(load_valY(), load_va()))
print(sklm.confusion_matrix(load_valY(), load_va()))
print(sklm.mean_squared_error(load_valY(), load_va()))
print(sklm.mean_absolute_error(load_valY(), load_va()))