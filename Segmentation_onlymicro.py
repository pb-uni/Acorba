"""
nbcs, Segmentation_onlymicro.py script.
Loaded buy the ACORBA GUI to perform microscopy root segmentation only with no
angle measurements.
"""
#Load libraries
import gc
import tifffile
import os
import cv2
from time import time,sleep
from tensorflow.autograph import set_verbosity
import numpy as np
import tifffile as tiff
from skimage.transform import resize
import PySimpleGUI as sg
import acorba

start_time = time()
set_verbosity(0)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
print("ALL LIBRARIES ARE IMPORTED, ACORBA WILL START NOW\n##################################################")

#retrieve user inputs from the GUI
args=acorba.folder.parser()

'''#for debugging
args.input_folder="E:/Pic libraries/test bank/Vincent/test2 new model/test v1.2/micro sand/"
saveplot="False"
rootplot="False"
exp_type="Microscopy Sandwich"
normalization="True"
predictiont="None"
seg_method="Deep Machine Learning"
args.binary_folder=""
custom_models=""
deactivate_smoothing="False"
os.chdir("E:/ACORBA/v1.2")
###'''

#Set random seed to "The Answer to the Ultimate Question of Life, The Universe, and Everything."
np.random.seed(42)
size=256

#Creates segmentation folder
dir = args.input_folder+"/Saved Segmentations/"
if not os.path.exists(dir):
    os.mkdir(dir)

#Load DML models and weights
if len(args.binary_folder)==0:
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
    
print("The following files will be processed")
print(rootlist)

#Setting the progress bar total length by opening every image lenght info and summing their timeframe numbers
looplenght=acorba.folder.looplenght_determinator(list_image=rootlist,folder=args.input_folder)

#Start for loop for all the file in the rootlist
for inc_root in rootlist:
    sg.theme('Light Brown 8')
    print("Processing file "+inc_root)
    test= tiff.imread(args.input_folder+'/'+inc_root)
    test_shape=test.shape
    tradlist=[]
    print("Converting stack to an array and resizing/padding to 256x256")
    X_test = np.zeros((len(test),size, size), dtype=np.float32)
    n=0
    while n<len(test):
        sg.OneLineProgressMeter(inc_root+' import, padding, resizing', n+1, len(test), 'key')
        im=(test[n]/255).astype('uint8')
        tradlist.append(im)
        if im.shape[0]!=im.shape[1]:
            desired_size = size
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
        n=n+1
        path=args.input_folder+'//Saved Segmentations/'+inc_root+'_'+'original_resized256.tif'
        tifffile.imwrite(path, X_test)
    tradlist=np.array(tradlist)
    #Predictions
    if args.method=="Deep Machine Learning":
        X_test=np.expand_dims(X_test,3)
        print("Prediction of root surface by deep machine learning")
        preds_test = loaded_model.predict(X_test, verbose=2,batch_size=1)
        prediction =(preds_test > 0.5).astype(np.uint8)
        print("Prediction of root tip by deep machine learning")
        preds_test = loaded_model_tip.predict(X_test, verbose=2,batch_size=1)
        prediction_tip =(preds_test > 0.5).astype(np.uint8)
    elif args.method=='Own masks':#create a macro to easily export binary masks
        prediction=tiff.imread(args.binary_folder+'/surface/'+inc_root)
        prediction_tip=tiff.imread(args.binary_folder+'/tip/'+inc_root)
        prediction= resize(prediction, (len(prediction),size, size), mode='constant', preserve_range=True)
        prediction_tip= resize(prediction_tip, (len(prediction_tip),size, size), mode='constant', preserve_range=True)
        if np.max(prediction>1):
            prediction=prediction/np.max(prediction)
        if np.max(prediction_tip>1):
            prediction_tip=prediction_tip/np.max(prediction_tip)
    else:
        print('Traditionnal segmentation of root surface')
        prediction=[]
        if args.exp_type=='Microscopy Through':
            for img_pred in X_test:
                prediction.append(acorba.microscope.trad_micro_through(img_pred.copy()))
        else:
            for img_pred in tradlist:
                prediction.append(acorba.microscope.trad_micro_sandwich(img_pred.copy()))
        prediction=np.array(prediction)
        X_test=np.expand_dims(X_test,3)
        print("Prediction of root tip by deep machine learning")
        preds_test = loaded_model_tip.predict(X_test, verbose=2,batch_size=1)
        prediction_tip =(preds_test > 0.5).astype(np.uint8)  
        
        
    path=args.input_folder+'//Saved Segmentations/'+inc_root+'_'+'root_tip_256.tif'
    tifffile.imwrite(path, prediction_tip)
    path=args.input_folder+'//Saved Segmentations/'+inc_root+'_'+'root_surface_256.tif'
    tifffile.imwrite(path, prediction)
    gc.collect()
    sleep(0.02)

print("I'm done")
print("it took me",time() - start_time, "seconds to do my job!")
print("Um9vdCByb2NrISB6bG9vcDI3")

gc.collect()
