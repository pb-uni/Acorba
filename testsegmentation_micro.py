# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 20:12:25 2021

@author: Nelson
"""

#Load libraries
import numpy as np
import os
import cv2
import gc
import tifffile as tiff
from skimage.transform import resize
from tensorflow.autograph import set_verbosity
import FreeSimpleGUI as sg
import acorba

set_verbosity(0)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
#retrieve user inputs from the GUI
args=acorba.folder.parser()

#for debugging
#args=acorba.debug.micro_debug()
#args.input_folder="E:/Pic libraries/test bank/Anastasiia/micro/"
#save_lenghts='False'

#Set random seed to "The Answer to the Ultimate Question of Life, The Universe, and Everything."
np.random.seed(42)
size=256

if len(args.custom)>0:
    from keras.models import model_from_json
    print('Custom models and weights loading:')
    model_weights=os.listdir(args.custom)
    custom_models_list=[]
    custom_weights_list=[]
    for i in model_weights:
        if i.endswith(".json"):
            custom_models_list.append(i)
        if i.endswith('.h5'):
            custom_weights_list.append(i)
    ans_surface = 'surface'
    ans_tip = 'tip'
    for title in custom_models_list:
        if ans_surface in title:
            with open(args.custom+"/"+title, 'r') as json_file:
                loaded_model_json = json_file.read()
            loaded_model = model_from_json(loaded_model_json)
            print('root surface model loaded')
        elif ans_tip in title:
            with open(args.custom+"/"+title, 'r') as json_file:
                loaded_model_json = json_file.read()
            loaded_model_tip = model_from_json(loaded_model_json)
            print('root tip model loaded')
    for title in custom_weights_list:
        if ans_surface in title:
            loaded_model.load_weights(args.custom+"/"+title)
            print('root surface weights loaded')
        elif ans_tip in title:
            loaded_model_tip.load_weights(args.custom+"/"+title)
            print('root tip weights loaded')
else:
    if args.exp_type=='Microscopy Through':
        from keras_unet.models import satellite_unet
        print("Default models and weights loading")
        loaded_model = satellite_unet(input_shape=(size, size, 1))
            #load root tip model
        loaded_model_tip = satellite_unet(input_shape=(size, size, 1))
            # load weights into new model
        loaded_model.load_weights("models/model_through_surface_256_sat_dice_update22092021_138.h5")
        loaded_model_tip.load_weights("models/model_through_tip_256_sat_dice_update22092021_193.h5")
    else:
        from keras_unet.models import custom_unet
        loaded_model=custom_unet(
            input_shape=(256, 256, 1),
            use_batch_norm=True,
            num_classes=1,
            filters=64,
            dropout=0.2,
            output_activation='sigmoid')
        loaded_model_tip=custom_unet(
            input_shape=(256, 256, 1),
            use_batch_norm=True,
            num_classes=1,
            filters=64,
            dropout=0.2,
            output_activation='sigmoid')
        loaded_model.load_weights("models/Microscopy_sandwich_256_dicecoef_customunet_01102021.h5")
        loaded_model_tip.load_weights("models/Microscopy_sandwich_TIP_256_dicecoef_customunet_01102021.h5")
print("Models and weights are loaded")
 
# Predict root masks
rootlist_temp=os.listdir(args.input_folder)
rootlist=[]
for i in rootlist_temp:
    if i.endswith(".tif"):
        rootlist.append(i)
del rootlist_temp

ntoimport=len(rootlist)

#Loop to retrieve all the first pictures
looplenght=0
desired_size = size
X_test = np.zeros((len(rootlist),size, size), dtype=np.float32)
n=0
tradlist=[]
sg.theme('Light Brown 8')
for inc_root in rootlist:
    sg.OneLineProgressMeter('Images import, padding, resizing', n+1, ntoimport, 'Image import')
    test=tiff.imread(args.input_folder+'/'+inc_root)
    test_shape=test.shape
    #check if test.dtype is 8 bits already
    im=(test[len(test)-1]/255).astype('uint8')
    tradlist.append(im)
    if im.shape[0]!=im.shape[1]:
        old_size = im.shape # old_size is in (height, width) format
        ratio = float(desired_size)/max(old_size)
        new_size = tuple([int(x*ratio) for x in old_size])
        img_t = cv2.resize(im, (new_size[1], new_size[0]))
        delta_w = desired_size - new_size[1]
        delta_h = desired_size - new_size[0]
        top, bottom = delta_h//2, delta_h-(delta_h//2)
        left, right = delta_w//2, delta_w-(delta_w//2)
        color = np.average(im)
        img_t = cv2.copyMakeBorder(img_t, top, bottom, left, right, cv2.BORDER_CONSTANT,value=color)
        X_test[n]=img_t/255
    else:
        X_test[n] = resize(im, (size, size), mode='constant', preserve_range=True)/255
    n+=1
#ML segmentation
X_test_ml=np.expand_dims(X_test,3)
print("Prediction of root surface")
preds_test = loaded_model.predict(X_test_ml, verbose=2,batch_size=1)
prediction =(preds_test > 0.5).astype(np.uint8)
print("Prediction of root tip")
preds_test = loaded_model_tip.predict(X_test_ml, verbose=2,batch_size=1)
prediction_tip =(preds_test > 0.5).astype(np.uint8)

#traditionnal segmentation
prediction_trad=[]
if args.exp_type=='Microscopy Through':
    for img in X_test:
        prediction_trad.append(acorba.microscope.trad_micro_through(img))
else:
    for image in tradlist:
        prediction_trad.append(acorba.microscope.trad_micro_sandwich(image))
    
prediction_trad=np.array(prediction_trad)


acorba.microscope.test_segmentations(X_test,prediction,prediction_tip,prediction_trad,args.input_folder,listnames=rootlist,nm_img_to_plot=len(X_test))
print("Segmentation test done!")
print("The segmentation test was saved in your experiment folder")
print("Um9vdCByb2NrISB6bG9vcDI3")
gc.collect()