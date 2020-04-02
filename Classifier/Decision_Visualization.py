#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = 8,8

from scipy.ndimage.interpolation import zoom
import VGG
import ResNet
import numpy as np
import os
import gradcamutils
from DenseNet import densenet
from PIL import Image

# use this environment flag to change which GPU to use 
os.environ["CUDA_VISIBLE_DEVICES"]="-1"  # specify which GPU(s) to be used

vggModel = VGG.VGG19((352,320,1),4) #set up model architecture
vggModel.summary()
vggModel.load_weights("/home/reza/DeepKneeExplainer/resources/old_models/balanced JSN/VGG19-JSNnewbalance-front-0.8896.h5") #load weights

denseNetModel = densenet.DenseNetImageNet161(input_shape=(352,320,1),classes=4, weights=None)
denseNetModel.summary()
denseNetModel.load_weights("/home/reza/DeepKneeExplainer/resources/old_models/balanced JSN/DenseNet161-JSNnewbalance-XRfront-0.8965.h5")

resNetModel = ResNet.ResNet34(input_shape=(352,320,1),classes=4)
resNetModel.summary()
resNetModel.load_weights("/home/reza/DeepKneeExplainer/resources/old_models/balanced JSN/ResNet34-JSNnewbalance-front-0.8395.h5")

img=Image.open("/home/reza/DeepKneeExplainer/resources/Data/XR/balancedXR ROI/front/validation/0082_R.png") #open image you want to visualize
img=np.array(img.resize((320,352), Image.ANTIALIAS))
im = img.reshape(1,352,320,1)
gradcam=gradcamutils.grad_cam(vggModel,im,layer_name='block5_conv4') #for VGG, here there are parameters to set image width (W) and height (H)
gradcamplus=gradcamutils.grad_cam_plus(vggModel,im,layer_name='block5_conv4')

fig, ax = plt.subplots(nrows=1,ncols=3)
plt.subplot(131)
plt.imshow(img, cmap ='gray')
plt.title("input image")
plt.subplot(132)
plt.imshow(img, cmap ='gray')
plt.imshow(gradcam,alpha=0.5,cmap="jet")
plt.title("Grad-CAM")
plt.subplot(133)
plt.imshow(img, cmap ='gray')
plt.imshow(gradcamplus,alpha=0.45,cmap="jet")
plt.title("Grad-CAM++")
#plt.show()

plt.savefig('VGGPlots.png')

plt.figure(figsize=(7,1))

probs = [0.12, 0.30, 0.48, 0.10]
probs = np.asarray(probs, dtype=np.float32)

objects = ('JSN0','JSN1','JSN2','JSN3')
y_pos = np.arange(len(objects))

for jsn in range(4):
    plt.text(jsn-0.2, 0.35, "%.2f" % np.round(probs[jsn],2), fontsize=10)

plt.bar(np.array([0, 1, 2, 3]), probs, color='red',align='center',tick_label=['JSN0','JSN1','JSN2','JSN3'],alpha=0.3)
plt.ylim(0,1)
plt.yticks([])
plt.xticks(y_pos, objects)
plt.savefig('VGGNetprob.png')

gradcam=gradcamutils.grad_cam(denseNetModel,im,layer_name='feature') #for DenseNet
gradcamplus=gradcamutils.grad_cam_plus(denseNetModel,im,layer_name='feature')

fig, ax = plt.subplots(nrows=1,ncols=3)
plt.subplot(131)
plt.imshow(img, cmap ='gray')
plt.title("input image")
plt.subplot(132)
plt.imshow(img, cmap ='gray')
plt.imshow(gradcam,alpha=0.5,cmap="jet")
plt.title("Grad-CAM")
plt.subplot(133)
plt.imshow(img, cmap ='gray')
plt.imshow(gradcamplus,alpha=0.45,cmap="jet")
plt.title("Grad-CAM++")

plt.savefig('DenseNetPlots.png')

plt.figure(figsize=(7,1))

probs = [0.03, 0.22, 0.51, 0.25]
probs = np.asarray(probs, dtype=np.float32)

for jsn in range(4):
    plt.text(jsn-0.2, 0.35, "%.2f" % np.round(probs[jsn],2), fontsize=10)

plt.bar(np.array([0, 1, 2, 3]), probs, color='red',align='center',tick_label=['JSN0','JSN1','JSN2','JSN3'],alpha=0.3)
plt.ylim(0,1)
plt.yticks([])
plt.xticks(y_pos, objects)
plt.savefig('DenseNetprob.png')

gradcam=gradcamutils.grad_cam(resNetModel,im,layer_name='conv2d_196') #for DenseNet
gradcamplus=gradcamutils.grad_cam_plus(resNetModel,im,layer_name='conv2d_196')

fig, ax = plt.subplots(nrows=1,ncols=3)
plt.subplot(131)
plt.imshow(img, cmap ='gray')
plt.title("input image")
plt.subplot(132)
plt.imshow(img, cmap ='gray')
plt.imshow(gradcam,alpha=0.5,cmap="jet")
plt.title("Grad-CAM")
plt.subplot(133)
plt.imshow(img, cmap ='gray')
plt.imshow(gradcamplus,alpha=0.45,cmap="jet")
plt.title("Grad-CAM++")

plt.savefig('ResNetPlots.png')

plt.figure(figsize=(7,1))

probs = [0.01, 0.25, 0.55, 0.10]
probs = np.asarray(probs, dtype=np.float32)

for jsn in range(4):
    plt.text(jsn-0.2, 0.35, "%.2f" % np.round(probs[jsn],2), fontsize=10)

plt.bar(np.array([0, 1, 2, 3]), probs, color='red',align='center',tick_label=['JSN0','JSN1','JSN2','JSN3'],alpha=0.3)
plt.ylim(0,1)
plt.yticks([])
plt.xticks(y_pos, objects)
plt.savefig('ResNetprob.png')