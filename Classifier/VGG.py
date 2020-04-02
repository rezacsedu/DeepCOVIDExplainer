from __future__ import absolute_import

from __future__ import division

from __future__ import print_function

import os
from keras import layers
from keras import models
from keras.models import *
from keras.layers import *


def VGG16():

    model = Sequential()

    model.add(ZeroPadding2D((1,1),input_shape=(544,352,1))) #here to change input shape

    model.add(Convolution2D(2, 3, 3, activation='relu',kernel_initializer="he_normal"))

    model.add(ZeroPadding2D((1,1)))

    model.add(Convolution2D(2, 3, 3, activation='relu',kernel_initializer="he_normal"))

    model.add(MaxPooling2D((2,2), strides=(2,2)))



    model.add(ZeroPadding2D((1,1)))

    model.add(Convolution2D(4, 3, 3, activation='relu',kernel_initializer="he_normal"))

    model.add(ZeroPadding2D((1,1)))

    model.add(Convolution2D(4, 3, 3, activation='relu',kernel_initializer="he_normal"))

    model.add(MaxPooling2D((2,2), strides=(2,2)))



    model.add(ZeroPadding2D((1,1)))

    model.add(Convolution2D(8, 3, 3, activation='relu',kernel_initializer="he_normal"))

    model.add(ZeroPadding2D((1,1)))

    model.add(Convolution2D(8, 3, 3, activation='relu',kernel_initializer="he_normal"))

    model.add(ZeroPadding2D((1,1)))

    model.add(Convolution2D(8, 3, 3, activation='relu',kernel_initializer="he_normal"))

    model.add(MaxPooling2D((2,2), strides=(2,2)))



    model.add(ZeroPadding2D((1,1)))

    model.add(Convolution2D(16, 3, 3, activation='relu',kernel_initializer="he_normal"))

    model.add(ZeroPadding2D((1,1)))

    model.add(Convolution2D(16, 3, 3, activation='relu',kernel_initializer="he_normal"))

    model.add(ZeroPadding2D((1,1)))

    model.add(Convolution2D(16, 3, 3, activation='relu',kernel_initializer="he_normal"))

    model.add(MaxPooling2D((2,2), strides=(2,2)))



    model.add(ZeroPadding2D((1,1)))

    model.add(Convolution2D(16, 3, 3, activation='relu',kernel_initializer="he_normal"))

    model.add(ZeroPadding2D((1,1)))

    model.add(Convolution2D(16, 3, 3, activation='relu',kernel_initializer="he_normal"))

    model.add(ZeroPadding2D((1,1)))

    model.add(Convolution2D(16, 3, 3, activation='relu',kernel_initializer="he_normal"))

    model.add(MaxPooling2D((2,2), strides=(2,2), name='inter'))



    model.add(Flatten())

    model.add(Dense(1024, activation='relu', name='fc1'))

    model.add(Dropout(0.5))

    model.add(Dense(256, activation='relu', name='fc2'))

    #model.add(Dropout(0.5))

    model.add(Dense(4, activation='softmax'))


    return model



def VGG19(input_shape=None,classes=5, use_soft=True):


    img_input = layers.Input(shape=input_shape)


    # Block 1

    x = layers.Conv2D(2, (3, 3),

                      activation='relu',

                      padding='same',

                      name='block1_conv1',kernel_initializer="he_normal")(img_input)

    x = layers.Conv2D(2, (3, 3),

                      activation='relu',

                      padding='same',

                      name='block1_conv2',kernel_initializer="he_normal")(x)

    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)



    # Block 2

    x = layers.Conv2D(4, (3, 3),

                      activation='relu',

                      padding='same',

                      name='block2_conv1',kernel_initializer="he_normal")(x)

    x = layers.Conv2D(4, (3, 3),

                      activation='relu',

                      padding='same',

                      name='block2_conv2',kernel_initializer="he_normal")(x)

    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)



    # Block 3

    x = layers.Conv2D(8, (3, 3),

                      activation='relu',

                      padding='same',

                      name='block3_conv1',kernel_initializer="he_normal")(x)

    x = layers.Conv2D(8, (3, 3),

                      activation='relu',

                      padding='same',

                      name='block3_conv2',kernel_initializer="he_normal")(x)

    x = layers.Conv2D(8, (3, 3),

                      activation='relu',

                      padding='same',

                      name='block3_conv3',kernel_initializer="he_normal")(x)
    x = layers.Conv2D(8, (3, 3),

                      activation='relu',

                      padding='same',

                      name='block3_conv4',kernel_initializer="he_normal")(x)

    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)



    # Block 4

    x = layers.Conv2D(16, (3, 3),

                      activation='relu',

                      padding='same',

                      name='block4_conv1',kernel_initializer="he_normal")(x)

    x = layers.Conv2D(16, (3, 3),

                      activation='relu',

                      padding='same',

                      name='block4_conv2',kernel_initializer="he_normal")(x)

    x = layers.Conv2D(16, (3,3),

                      activation='relu',

                      padding='same',

                      name='block4_conv3',kernel_initializer="he_normal")(x)
    x = layers.Conv2D(16, (3,3),

                      activation='relu',

                      padding='same',

                      name='block4_conv4',kernel_initializer="he_normal")(x)

    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)



    # Block 5

    x = layers.Conv2D(16, (3, 3),

                      activation='relu',

                      padding='same',

                      name='block5_conv1',kernel_initializer="he_normal")(x)

    x = layers.Conv2D(16, (3, 3),

                      activation='relu',

                      padding='same',

                      name='block5_conv2',kernel_initializer="he_normal")(x)

    x = layers.Conv2D(16, (3, 3),

                      activation='relu',

                      padding='same',

                      name='block5_conv3',kernel_initializer="he_normal")(x)
    x = layers.Conv2D(16, (3, 3),

                      activation='relu',

                      padding='same',

                      name='block5_conv4',kernel_initializer="he_normal")(x)

    x = layers.MaxPooling2D((2, 2), strides=(2,2), name='block5_pool')(x)


        # Classification block

    x = layers.Flatten(name='flatten')(x)

    x = layers.Dense(512, activation='relu', name='fc1')(x)
    x=layers.Dropout(0.5)(x)

    x = layers.Dense(128, activation='relu', name='fc2')(x)
    #x=layers.Dropout(0.8)(x)

    if use_soft:
        x = Dense(classes, activation = "softmax", name='predictions')(x)
    else:
        x = Dense(classes, activation = "linear", name = "Z_4")(x)

    model = models.Model(img_input, x, name='vgg19')

    return model

def VGG19_dense(input_shape=None,classes=4):


    img_input = layers.Input(shape=input_shape)


    # Block 1

    x = layers.Conv2D(2, (3, 3),

                      activation='relu',

                      padding='same',

                      name='block1_conv1',kernel_initializer="he_normal")(img_input)

    x = layers.Conv2D(2, (3, 3),

                      activation='relu',

                      padding='same',

                      name='block1_conv2',kernel_initializer="he_normal")(x)

    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)



    # Block 2

    x = layers.Conv2D(4, (3, 3),

                      activation='relu',

                      padding='same',

                      name='block2_conv1',kernel_initializer="he_normal")(x)

    x = layers.Conv2D(4, (3, 3),

                      activation='relu',

                      padding='same',

                      name='block2_conv2',kernel_initializer="he_normal")(x)

    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)



    # Block 3

    x = layers.Conv2D(8, (3, 3),

                      activation='relu',

                      padding='same',

                      name='block3_conv1',kernel_initializer="he_normal")(x)

    x = layers.Conv2D(8, (3, 3),

                      activation='relu',

                      padding='same',

                      name='block3_conv2',kernel_initializer="he_normal")(x)

    x = layers.Conv2D(8, (3, 3),

                      activation='relu',

                      padding='same',

                      name='block3_conv3',kernel_initializer="he_normal")(x)
    x = layers.Conv2D(8, (3, 3),

                      activation='relu',

                      padding='same',

                      name='block3_conv4',kernel_initializer="he_normal")(x)

    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)



    # Block 4

    x = layers.Conv2D(16, (3, 3),

                      activation='relu',

                      padding='same',

                      name='block4_conv1',kernel_initializer="he_normal")(x)

    x = layers.Conv2D(16, (3, 3),

                      activation='relu',

                      padding='same',

                      name='block4_conv2',kernel_initializer="he_normal")(x)

    x = layers.Conv2D(16, (3,3),

                      activation='relu',

                      padding='same',

                      name='block4_conv3',kernel_initializer="he_normal")(x)
    x = layers.Conv2D(16, (3,3),

                      activation='relu',

                      padding='same',

                      name='block4_conv4',kernel_initializer="he_normal")(x)

    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)



    # Block 5

    x = layers.Conv2D(16, (3, 3),

                      activation='relu',

                      padding='same',

                      name='block5_conv1',kernel_initializer="he_normal")(x)

    x = layers.Conv2D(16, (3, 3),

                      activation='relu',

                      padding='same',

                      name='block5_conv2',kernel_initializer="he_normal")(x)

    x = layers.Conv2D(16, (3, 3),

                      activation='relu',

                      padding='same',

                      name='block5_conv3',kernel_initializer="he_normal")(x)
    x = layers.Conv2D(16, (3, 3),

                      activation='relu',

                      padding='same',

                      name='block5_conv4',kernel_initializer="he_normal")(x)

    x = layers.MaxPooling2D((2, 2), strides=(2,2), name='block5_pool')(x)


    x = layers.Conv2D(32, (3, 3), activation='relu',name="feature")(x)
    x = layers.Conv2D(32, (1, 1), activation='relu')(x)
    x = layers.Conv2D(classes, (1, 1), activation='softmax')(x)
    x = layers.AveragePooling2D((9,9))(x)

    model = models.Model(img_input, x, name='vgg19')
    model.summary()



    return model

