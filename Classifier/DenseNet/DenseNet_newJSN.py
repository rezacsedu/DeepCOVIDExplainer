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
  reader = open("/data/jiao/newlabel.csv") #label file path
  data=reader.readlines()
  files = os.listdir('/data/jiao/XR/ROI_resize/front/training/') #training path for ROIs/MRIs
  shuffle(files)
  for file in files:
        if file.endswith(".xml"):continue
        fi_d = os.path.join('/data/jiao/XR/ROI_resize/front/training/',file) #training path for ROIs/MRIs
        img=Image.open(fi_d).convert('L')
        im=np.array(img.resize((320,352), Image.ANTIALIAS))
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
        if "V" in file:  #for dataset balance, I use Grade 3 images from other stages, they are named with stagename as V3
                       label="3"
        if "8" not in label and "9" not in label and "X" not in label and '.' not in label: #in the labels, there are 8, 9 and X which are useless in our case.
          #if "." in label:
            #label='4'
          label= dense_to_one_hot(int(label),4)
          imgList.append(im)
          labelList.append(label)
  return np.array(imgList),np.array(labelList)

def load_val():
  imgList=[]
  labelList=[]
  reader = open("/data/jiao/newlabel.csv") #label file path
  data=reader.readlines()
  files = os.listdir('/data/jiao/XR/ROI_resize/front/validation/') #test path for ROIs/MRIs
  for file in files:
        if file.endswith(".xml"):continue
        fi_d = os.path.join('/data/jiao/XR/ROI_resize/front/validation/',file) #test path for ROIs/MRIs
        img=Image.open(fi_d).convert('L')
        im=np.array(img.resize((320,352), Image.ANTIALIAS))
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
          label= dense_to_one_hot(int(label),4)
          imgList.append(im)
          labelList.append(label)
  return np.array(imgList),np.array(labelList)

def load_valY(): #load labels with decimal format
  imgList=[]
  labelList=[]
  reader = open("/data/jiao/newlabel.csv")
  data=reader.readlines()
  files = os.listdir('/data/jiao/XR/ROI_resize/front/validation/')
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


batch_size=32
model = densenet.DenseNetImageNet201(input_shape=(352,320,1),classes=4, weights=None) #here you can change Densenet for 121,161,169 and 201 or your own architectures, the detail settings are input_shape=None, bottleneck=True,reduction=0.5, dropout_rate=0.0,weight_decay=1e-6,include_top=True, weights='imagenet',input_tensor=None,classes=1000, activation='softmax'
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.95, nesterov=True)
model.compile(optimizer=sgd, loss='mse',metrics=['accuracy'])

datagen = ImageDataGenerator(
        featurewise_center=True, 
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=True,  
        samplewise_std_normalization=False)  
X_train, Y_train = load()
X_test, Y_test = load_val()
X_train = X_train.reshape( len(X_train), len(X_train[0]), len(X_train[0][0]),1)
X_test = X_test.reshape( len(X_test), len(X_test[0]), len(X_test[0][0]),1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
datagen.fit(X_train) 
for i in range(len(X_test)):
      X_test[i] = datagen.standardize(X_test[i])
earlystop=EarlyStopping(monitor='val_acc', min_delta=0, patience=300, verbose=1, mode='auto', restore_best_weights=True)       
history = model.fit_generator(datagen.flow(X_train, Y_train,batch_size=batch_size),steps_per_epoch=32,epochs=4096,shuffle=True,validation_data=(X_test, Y_test), verbose=1,callbacks=[earlystop])
score, acc = model.evaluate(X_test,Y_test,batch_size=batch_size)
print("Accuracy:",acc)
if acc>0.6: #if the accuracy is higher than 60%, the models are saved
    model.save_weights("DenseNet-JSNnew-front.h5")
y_pred = model.predict(X_test)
Y_predict = y_pred.argmax(axis=-1)
f=open('DenseNetRESULTS-JSNnew-front.txt','a') #create performance report
f.write(classification_report(load_valY(), Y_predict))
f.write(str(sklm.cohen_kappa_score(load_valY(), Y_predict))+","+str(acc)+","+str(score)+"\n")
print(classification_report(load_valY(), Y_predict))
