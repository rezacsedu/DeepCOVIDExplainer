from __future__ import print_function
import numpy as np

np.random.seed(3768)  # for reproducibility
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential,load_model,Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import *
from keras.optimizers import SGD
from random import shuffle
import time
import csv
import os
import VGG
import densenet
from keras.callbacks import CSVLogger
from keras import callbacks
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import keras
from sklearn.metrics import classification_report
import sklearn.metrics as sklm
from keras.callbacks import EarlyStopping
from keras.callbacks import LearningRateScheduler
from keras import initializers
import keras
import tensorflow as tf

labelpath="/data/jiao/newlabel.csv"
trainingpath="/data/jiao/MRI/up/training/"
testpath="/data/jiao/MRI/up/validation/"

def get_session(): 
  config = tf.ConfigProto() 
  config.gpu_options.allow_growth = True 
  return tf.Session(config=config) 
# use this environment flag to change which GPU to use 
#os.environ["CUDA_VISIBLE_DEVICES"] = "" 
# set the modified tf session as backend in keras 
keras.backend.tensorflow_backend.set_session(get_session())
  
def dense_to_one_hot(labels_dense,num_clases=5):
  return np.eye(num_clases)[labels_dense]

def load():
  imgList=[]
  labelList=[]
  nameList=[]
  reader = open(labelpath)
  data=reader.readlines()
  files = os.listdir(trainingpath)
  shuffle(files)
  for file in files:
        if file.endswith(".xml"):continue
        fi_d = os.path.join(trainingpath,file)
        img=Image.open(fi_d).convert('L')
        im=np.array(img.resize((360,360), Image.ANTIALIAS))
        patient=file.split('_')[0]
        direction=file.split('_')[1].split('.')[0]
        naming=file.split('.')[0]
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
          label= dense_to_one_hot(int(label),4)
          imgList.append(im)
          labelList.append(label)
          nameList.append(naming)
  return np.array(imgList),np.array(labelList),np.array(nameList)

def load_val():
  imgList=[]
  labelList=[]
  nameList=[]
  reader = open(labelpath)
  data=reader.readlines()
  files = os.listdir(testpath)
  for file in files:
        if file.endswith(".xml"):continue
        fi_d = os.path.join(testpath,file)
        img=Image.open(fi_d).convert('L')
        im=np.array(img.resize((360,360), Image.ANTIALIAS))
        patient=file.split('_')[0]
        direction=file.split('_')[1].split('.')[0]
        naming=file.split('.')[0]
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
          label= dense_to_one_hot(int(label),4)
          imgList.append(im)
          labelList.append(label)
          nameList.append(naming)
  return np.array(imgList),np.array(labelList),np.array(nameList)

def load_valY():
  imgList=[]
  labelList=[]
  nameList=[]
  reader = open(labelpath)
  data=reader.readlines()
  files = os.listdir(testpath)
  for file in files:
        if file.endswith(".xml"):continue
        patient=file.split('_')[0]
        direction=file.split('_')[1].split('.')[0]
        naming=file.split('.')[0]
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
          nameList.append(naming)
  return np.array(labelList)

batch_size=32
model = VGG.VGG19_dense((360,360,1),4) #here you can change VGG19/VGG16/revised VGG19
model.load_weights("VGG19-JSNnewbalance-MRIup-0.6536.h5")  #load trained models

datagen = ImageDataGenerator(
        featurewise_center=True, 
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=True,  
        samplewise_std_normalization=False)  
X_train, Y_train, N_train = load()
X_test, Y_test, N_test = load_val()
X_train = X_train.reshape( len(X_train), len(X_train[0]), len(X_train[0][0]),1)
X_test = X_test.reshape( len(X_test), len(X_test[0]), len(X_test[0][0]),1)
Y_train = Y_train.reshape( len(Y_train), 1, 1,4)
Y_test = Y_test.reshape( len(Y_test), 1, 1,4)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
datagen.fit(X_train) 
for i in range(len(X_test)):
      X_test[i] = datagen.standardize(X_test[i])
y_pred = model.predict(X_test)
y_pred = y_pred.reshape( len(Y_test), 4)
Y_predict = y_pred.argmax(axis=-1)
f=open('VGG-MRIup-pos1.csv','a')
for i in range(len(y_pred)):
     f.write(N_test[i]+","+str(y_pred[i])+"\n")
