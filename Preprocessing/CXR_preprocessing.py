import numpy as np
from numpy import *
import os
import optparse
from PIL import Image
import png
import pylab
from scipy import misc, ndimage
from medpy.filter.smoothing import anisotropic_diffusion
from sklearn import preprocessing

def histogram_t(tb):
    '''
    Histogram equalization
    '''
    totalpixel=0    
    maptb=[]        
    count=len(tb)
    for i in range(count):
        totalpixel+=tb[i]
        maptb.append(totalpixel)

    for i in range(count):
        maptb[i]=int(round((maptb[i]*(count-1))/totalpixel))
   
    def histogram(light):
        return maptb[light]
    return histogram

def remove_annotation(img):
    '''
    Remove textual artifacts from X-ray imagese.g., a large number of images indicate the right side of the chest with a white `R' character, `L' for left.
    :param img: Numpy array of image
    :return: Array of image with possible characters removed and inpainted. 
    '''
    mask = cv2.threshold(img, 224, 224, cv2.THRESH_BINARY)[1][:, :, 0].astype(np.uint8)
    img = img.astype(np.uint8)
    result = cv2.inpaint(img, mask, 10, cv2.INPAINT_NS).astype(np.float32)
    return result

imagepath='cleanedCXR/' #image folder path, please pay attention to here images are already renamed with format "patient_direction". 
files = os.listdir(imagepath)

for fi in files:
  fi_d = os.path.join(imagepath,fi)
  img=Image.open(fi_d)
  
  if 'P' in fi:  #for chest x-rays
    (tempx1,tempy1)=img.size #original image separation
    width=tempx1//2
    left=img.crop((0,0,width,tempy1))
    right=img.crop((width,0,tempx1,tempy1))
    imgl = Image.new('RGBA',(width,tempy1))
    imgr = Image.new('RGBA',(width,tempy1))
    imgl.paste(left)
    imgr.paste(right)
    imgr=imgr.transpose(Image.FLIP_LEFT_RIGHT) #image flipping
    outl=imgl.resize((1023,2047),Image.ANTIALIAS) #image resize
    outr=imgr.resize((1023,2047),Image.ANTIALIAS)
    hisl=outl.histogram() #histogram
    hisfuncl=histogram_t(hisl) 
    iml=outl.point(hisfuncl)  
	
    hisr=outr.histogram() 
    hisfuncr=histogram_t(hisr) 
    imr=outr.point(hisfuncr)    
	
    ir = anisotropic_diffusion(np.array(imr)) #noise removal
    ir = remove_annotation(ir)
	
    il = anisotropic_diffusion(np.array(iml))	
    il = remove_annotation(il)
	
    temp='cleanedCXR/temp/'+fi[0:4] #processed image path for saving
    imagel = Image.fromarray(il.astype('uint8')).convert("L")
    imager = Image.fromarray(ir.astype('uint8')).convert("L") 
	
    imager.save(temp+'_PR.png')
