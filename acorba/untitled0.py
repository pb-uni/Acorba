# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 23:53:51 2022

@author: Nelson
"""

from skimage.io import imread,imshow
import cv2
import numpy as np
import tifffile
a=imread("E:/Pic libraries/test bank/Vincent/test2 new model/test v1.2/Scanner/4tf/Stacks (1)-1.tif")
a.shape
lene=a
if min(lene.shape)==lene.shape[2]:
    lene=np.rollaxis(lene,2,0)

lene.shape

path="E:/Pic libraries/test bank/Vincent/test2 new model/test v1.2/Scanner/4tf/test.tif"
path2="E:/Pic libraries/test bank/Vincent/test2 new model/test v1.2/Scanner/4tf/test2.tif"
b=imread("C:/TEMP to sup/test len vgi hgi/1persucrose.tif")
b.shape
tifffile.imwrite(path, lene,imagej=True,metadata={'axes': 'TYX'},shape=(lene.shape))
tifffile.imwrite(path2, b, shape=(b.shape),metadata={'axes': 'TYX'})
