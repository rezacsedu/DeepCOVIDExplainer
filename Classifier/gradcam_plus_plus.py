import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = 8,8

from scipy.ndimage.interpolation import zoom
import VGG
import densenet
import numpy as np
import os
import gradcamutils
from PIL import Image

model=VGG.VGG19_1((352,320,1),4) #set up model architecture
#model = densenet.DenseNetImageNet161(input_shape=(352,320,1),classes=4, weights=None)
#model.load_weights("DenseNet-JSNnewbalance-XRfront-0.8965.h5")
model.load_weights("VGG19-JSNnewbalance-front-0.8896.h5") #load weights
img=Image.open("T:/ROI_resize/front/validation/0082_R.png") #open image you want to visualize
img=np.array(img.resize((320,352), Image.ANTIALIAS))
im = img.reshape( 1,352,320,1)
gradcam=gradcamutils.grad_cam(model,im,layer_name='block5_conv4') #for VGG, here there are parameters to set image width (W) and height (H)
gradcamplus=gradcamutils.grad_cam_plus(model,im,layer_name='block5_conv4')
#gradcam=gradcamutils.grad_cam(model,im,layer_name='feature') #for DenseNet
#gradcamplus=gradcamutils.grad_cam_plus(model,im,layer_name='feature')
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
plt.show()
