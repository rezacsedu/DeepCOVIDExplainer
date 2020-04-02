from __future__ import print_function

import numpy as np
from random import shuffle
import time
import csv
from PIL import Image
import os
import tensorflow as tf
import keras
from keras.callbacks import EarlyStopping, LearningRateScheduler
from keras import initializers
from keras.optimizers import SGD
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential,load_model,Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import *
from keras.callbacks import CSVLogger
from keras import callbacks
from keras.preprocessing.image import ImageDataGenerator

import networks.VGG as vgg
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import sklearn.metrics as sklm
from utils.lossprettifier import LossPrettifier

# for reproducibility
np.random.seed(3768)

# use this environment flag to change which GPU to use 
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"  # specify which GPU(s) to be used

#Get TensorFlow session
def get_session(): 
  config = tf.ConfigProto() 
  config.gpu_options.allow_growth = True 
  return tf.Session(config=config) 
  
# One hot encoding of labels 
def dense_to_one_hot(labels_dense,num_clases=5):
  return np.eye(num_clases)[labels_dense]

MRI_LabelPath = '/home/reza/DeepKneeExplainer/resources/newlabel.csv'
train_MRI_image_Path = '/home/reza/DeepKneeExplainer/resources/Data/MRI/unbalanced/up/training'
test_MRI_image_Path = '/home/reza/DeepKneeExplainer/resources/Data/MRI/unbalanced/up/validation'
modelPath = "VGG19-KLnew.h5"
resultPath = 'VGG19_RESULTS-JSNnewbalance-MRIup.txt'

# Loading training MRI images and labels
def loadTrainImageAndLabels():
  imgList = []
  labelList = []
  readLabels = open(MRI_LabelPath) #label file path
  data = readLabels.readlines()
  
  files = os.listdir(train_MRI_image_Path) #training path
  shuffle(files)
  
  for file in files:
        if file.endswith(".xml"):continue
        fi_d = os.path.join(train_MRI_image_Path, file) #training path
        img = Image.open(fi_d).convert('L')

        im = np.array(img.resize((360,360), Image.ANTIALIAS))
        patient = file.split('_')[0]
        direction = file.split('_')[1].split('.')[0]
        label = "q"

        for row in data:
           if patient in row.split(",")[0]:
              if "L" in direction:
                 label=row.split(",")[1]
              else:
                 label=row.split(",")[4]
              break
        if "8" not in label and "9" not in label:
          label= dense_to_one_hot(int(label),5)
          imgList.append(im)
          labelList.append(label)

  return np.array(imgList),np.array(labelList)

# Loading validation MRI images
def loadTestImagesAndLabels():
  imgList = []
  labelListOneHot = []
  labelList = []
  readLabels = open(MRI_LabelPath) #label file path
  data = readLabels.readlines()
  files = os.listdir(test_MRI_image_Path) #test path

  for file in files:
        if file.endswith(".xml"):continue
        fi_d = os.path.join(test_MRI_image_Path, file) #test path
        
        img = Image.open(fi_d).convert('L')
        im = np.array(img.resize((360,360), Image.ANTIALIAS))
        patient = file.split('_')[0]
        direction = file.split('_')[1].split('.')[0]
        label = "q"
        
        for row in data:
           if patient in row.split(",")[0]:
              if "L" in direction:
                 label=row.split(",")[1]
              else:
                 label=row.split(",")[4]
              break
        if "8" not in label and "9" not in label:
          labelOneHot = dense_to_one_hot(int(label),5)
          imgList.append(im)

          labelListOneHot.append(labelOneHot)
          labelList.append(int(label))

  return np.array(imgList), np.array(labelListOneHot), np.array(labelList)

#Defining hyperparameters
batch_Size = 32
steps_Per_Epoch = 32
numEpochs = 20

#Instantating VGG19 model
model = vgg.VGG19_dense((360,360,1),5) #VGG19_dense for revised VGG19, VGG19 for VGG19. Please pay attention to VGG16(), chnage the input shape and class number in VGG.py.

#Creating an optimizers
adaDelta = keras.optimizers.Adadelta(lr=1.0, rho=0.95)
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.95, nesterov=True)
model.compile(optimizer = sgd , loss = 'categorical_crossentropy', metrics = ['accuracy'])

#Image data generation for the training 
datagen = ImageDataGenerator(
               featurewise_center = False, 
               samplewise_center = False,  # set each sample mean to 0
               featurewise_std_normalization = True,  
               samplewise_std_normalization = False)  

# Preparing training and test sets
X_train, Y_train = loadTrainImageAndLabels()
x_train, x_valid, y_train, y_valid = train_test_split(X_train, Y_train, test_size=0.20, random_state=42)

y_train = y_train.reshape(len(y_train), 1, 1,5)
y_valid= y_valid.reshape(len(y_valid), 1, 1,5)

x_train = x_train.reshape(len(x_train), len(x_train[0]), len(x_train[0][0]),1)
x_train = x_train.astype('float32')
x_train /= 255

x_valid= x_valid.reshape(len(x_valid), len(x_valid[0]), len(x_valid[0][0]),1)
x_valid= x_valid.astype('float32')
x_valid/= 255

x_test, y_test_oneHot, y_test = loadTestImagesAndLabels()
y_test_oneHot = y_test_oneHot.reshape(len(y_test_oneHot), 1, 1,5)

x_test = x_test.reshape(len(x_test), len(x_test[0]), len(x_test[0][0]),1)
x_test = x_test.astype('float32')
x_test /= 255

datagen.fit(x_train) 
for i in range(len(x_test)):
      x_test[i] = datagen.standardize(x_test[i])

#Creating early stopping 
earlystop = EarlyStopping(monitor = 'val_accuracy', min_delta = 0, patience = 50, verbose = 1, mode = 'auto', restore_best_weights = True)       

train_generator = datagen.flow(x_train, y_train, batch_size = batch_Size)
validation_generator = datagen.flow(x_valid, y_valid, batch_size = batch_Size)

# Model training
history = model.fit_generator(
    train_generator,
    steps_per_epoch = steps_Per_Epoch,
    validation_data = validation_generator, 
    validation_steps = 16,
    epochs = numEpochs,
    shuffle = True, 
    verbose = 1)

# visualizing losses and accuracy
train_loss = history.history['loss']
val_loss = history.history['val_loss']

#Observing the losses but can be commented out as it's not mandatory 
reporter = LossPrettifier(show_percentage=True)
for i in range(numEpochs-1):
    reporter(epoch=i, LossA = train_loss[i], LossAB = val_loss[i])

# Model evaluation 
score, acc = model.evaluate(x_test, y_test_oneHot, batch_size=batch_Size)
print("Accuracy:", acc)

#if acc>0.675:
model.save_weights(modelPath)

y_pred = model.predict(x_test)
y_pred = y_pred.reshape(len(y_test_oneHot), 5)
y_predict = y_pred.argmax(axis=-1)

# Writing results on file
f = open(resultPath,'a') #create classification report
f.write(classification_report(y_test, y_predict))
f.write(str(sklm.cohen_kappa_score(y_test, y_predict))+","+str(acc)+","+str(score)+"\n")

#Print class-wise classification metrics
print(classification_report(y_test, y_predict))
