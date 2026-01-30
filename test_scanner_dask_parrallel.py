# -*- coding: utf-8 -*-
"""
Created on Fri Nov  5 19:41:19 2021

@author: Nelson
"""

'''
nbcs, scanner.py script.
Loaded buy the ACORBA GUI to perform scanner root segmentation and angle
measurements.
'''
#Load dependencies
import os
import sys
import gc
import acorba
from time import sleep, time
import tifffile
import cv2
from skimage.io import imread
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import FreeSimpleGUI as sg
import dask
sg.theme('Light Brown 8')
start_time = time()#Start measuring time, so the user knows how long took his session
#To silence non-essential tensorflow informations and non-consequential warnings


print("ALL LIBRARIES ARE IMPORTED, ACORBA WILL START NOW\
      n##################################################")

plt.ioff()#Shuts off matplotlib interactive mode

#retrieve user inputs from the GUI
#For debugging
"""
args=acorba.debug.scanner_debug()
"""
#retrieve user parameter from the parser
args=acorba.folder.parser()

#Set random seed to "The Answer to the Ultimate Question of Life, The Universe, and Everything."
np.random.seed(42)

#Set the broken skeleton repair treshold (in pixels)
#if two skeleton end and origin are spaced < 100 pixels they will be stitched
brokenfactor=100

#set the number of pixel to set the origin of the angle vector
originvector=10

#Create a folder for segmentation tif stack in case it was ticked by the user
if args.savesegmentation=="True":
    exp_dir = args.input_folder+"/Saved Segmentations/"
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)

#Retrieve images list
listimage=acorba.folder.list_tif(folder=args.input_folder)

stack2=0#Increment for the progress bar

#Load Deep Machine Learning libraries and models/weights
if args.method=='Deep Machine Learning':
    from tensorflow.autograph import set_verbosity
    set_verbosity(0)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    if len(args.custom)>0:
        from keras.models import model_from_json
        print('Custom models and weights loading:')
        model_weights=os.listdir(args.custom)
        for i in model_weights:
            if i.endswith(".json"):
                with open(args.custom+"/"+i, 'r') as json:#check if its working
                    loaded_model_json = json.read()
                model = model_from_json(loaded_model_json)
            if i.endswith('.h5'):
                model.load_weights(args.custom+"/"+i)
    else:
        from keras_unet.models import satellite_unet
        model = satellite_unet(input_shape=(256, 256, 1))
        if args.superaccuracy=="True":
            model.load_weights("models/unetnocontrast_lessaugm_09092021_22.h5")
        else:
            model.load_weights("models/unetnocontrast_lessaugm_09092021_87.h5")
    print("Scanner prediction model and weights loaded")

#Setting the progress bar total length by opening every image lenght info and summing their timeframe numbers
looplenght=acorba.folder.looplenght_determinator(
    list_image=listimage,folder=args.input_folder)


listimage2=[]
#Starting analysis
output=[]
for images in listimage:#loop for .tif stacks
    a=full_scanner(images,args,looplenght,model,brokenfactor,stack2,listimage2,originvector)
    output.append(a)
total = dask.delayed()(output)
start=time()
total.compute()
end=time()-start
            #retrieve non processed image list
listremoved=listimage2
plt.close()
print("I'm done")
if len(listremoved)==0:
    print("everything went well")
else:
    print("The following files raised errors and were not analyzed fully: ",listremoved)
print("it took me",time() - start_time, "seconds to do my job!")
print("Um9vdCByb2NrISB6bG9vcDI3")
#collect garbage files
gc.collect()
