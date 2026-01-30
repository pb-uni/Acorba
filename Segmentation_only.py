"""
nbcs, Segmentation_only.py script.
Loaded buy the ACORBA GUI to perform scanner root segmentation only with no
angle measurements.
"""

#Load dependencies
import sys
import os
import gc
from time import sleep, time
import tifffile
from skimage.io import imread
import matplotlib.pyplot as plt
import numpy as np
import FreeSimpleGUI as sg
from tensorflow.autograph import set_verbosity
import acorba

sg.theme('Light Brown 8')
#Start measuring time, so the user knows how long took his session
start_time = time()
#Silence non-essential tensorflow infos and non-consequential warnings
set_verbosity(0)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

print("ALL LIBRARIES ARE IMPORTED, ACORBA WILL START NOW\
      n##################################################")

plt.ioff()#Shuts off matplotlib interactive mode

#retrieve user inputs from the GUI
args=acorba.folder.parser()
#Debug
#args=acorba.debug.scanner_debug()

#Set random seed to "The Answer to the Ultimate Question of Life, The Universe,
#and Everything
np.random.seed(42)

#if two skeleton end and origin are spaced < 100 pixels they will be stitched
brokenfactor=int(args.broken)

#set the number of pixel to set the origin of the angle vector
originvector=int(args.vector)



#Retrieve images list
listimage_temp= os.listdir(args.input_folder)
listimage = [i for i in listimage_temp if i.endswith(".tif")]
listimage.sort()
print("The following image will be processed:")
print(listimage)
del listimage_temp #clean temporary list

#Creates a directory to save the segmentations separetely from the originals
#Helps when someone wants to run an analysis after
exp_dir = args.input_folder+"/Saved Segmentations/"
if not os.path.exists(exp_dir):
    os.mkdir(exp_dir)

stack2=0#Increment for the progress bar
#Load Deep Machine Learning libraries and models/weights
if args.method=='Deep Machine Learning':
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
            model.load_weights(
                "models/unetnocontrast_lessaugm_09092021_22.h5")
        else:
            model.load_weights(
                "models/unetnocontrast_lessaugm_09092021_87.h5")
    print("Scanner prediction model and weights loaded")

#Setting the progress bar total length by opening every image lenght info and
#summing their timeframe numbers
looplenght=acorba.folder.looplenght_determinator(list_image=listimage,folder=args.input_folder)

listimage2=[]
#Starting analysis
for images in listimage:#loop for .tif stacks
    seg_list=[]#create segmentation list to transform into an array at the end
    #and imwrite, save padded picture if machine learning regular
    if args.method=="Deep Machine Learning" and args.superaccuracy=="False":
        im_list=[]
    if not sg.OneLineProgressMeter(
            'Segmentation in progress', stack2, looplenght, 'single'):
        sys.exit("User stop!!!!!!!!!!!!!!")
    lene=imread(args.input_folder+'/'+images)#Stack import
    if min(lene.shape)==lene.shape[2]:
        lene=np.rollaxis(lene,2,0)
    ntf=int(min(lene.shape))#Finding the number of timeframes
    imgname=images#Store image name in a different object as images part of
    #the for loop it was creating problem with the try/except functions
    if args.prediction in ("First",'All'):
        #Create empty list for prediction export
        predsexp=[]
        imgsexp=[]
    imnb=0
    angletot=[]
    while imnb < ntf:#Loop for timeframe
        im=lene[imnb]
        if args.method=="Deep Machine Learning":
            if args.superaccuracy=="False":
                from patchify import unpatchify
                #Padding the image to a int of 256x256 tiles and tiling it
                new_im, X, factorv, factorh=acorba.scanner.padding_tiling(
                    image_input=im, tile=256)

                #Preprocessing of array for fitting the model data input
                X=np.array(X)#transform X from a list to an array
                X=X/255#Value between 0 and 1: data normalization
                X=np.expand_dims(X,3)#Model input is (256,256,1)

                #Predictions
                preds_train = model.predict(X, verbose=2,batch_size=1)
                preds_train_t = (preds_train > 0.5).astype(np.uint8)

                #Reconstruction of a segmentation image from the tiles
                #architecture of the patching
                preds_train_t.shape=(factorv,factorh,256,256)
                #patching
                mega=unpatchify(preds_train_t, new_im.shape)
                #Super accuracy mode for scanner >> smooth tiling, requires
                #tons of processing power and RAM!
                im_list.append(new_im)
            else:
                im_expanded=np.expand_dims(im,2)#input is (256,256,1)
                im_expanded=im_expanded/255#data normalization between 0 and 1
                predictions_smooth = acorba.scanner.predict_img_with_smooth_windowing(
                    input_img=im_expanded,
                    window_size=256,
                    # Minimal amount of overlap for windowing.
                    #Must be an even number.
                    subdivisions=2,
                    nb_classes=1,
                    pred_func=(
                        lambda img_batch_subdiv:
                            model.predict(
                                (img_batch_subdiv), verbose=2,batch_size=1)
                    )
                )
#somehow the non-root is under 0.01 instead of 0.5 for non smooth prediction
                mega = (predictions_smooth > 0.01).astype(np.uint8)
                new_im=im
        elif args.method=="Traditional":
            if args.tradmethod=="Entropy":
                mega=acorba.scanner.trad_scanner(im)
            else:
                mega=acorba.scanner.trad_scanner_threshold(im)
        elif args.method=="Own masks":
            mega=imread(args.binary_folder+'/'+images,key=imnb)
            if np.max(mega>1):
                mega=mega/np.max(mega)
            mega=mega.astype('uint8')
        seg_list.append(mega)
        imnb=imnb+1
        stack2=stack2+1
    seg_list=np.array(seg_list)
    if args.superaccuracy=="True":
        seg_list=np.squeeze(seg_list, axis=(3,))
    path=args.input_folder+'//Saved Segmentations/'+images+'_'+'segmented.tif'
    tifffile.imwrite(path, seg_list,imagej=True,metadata={'axes': 'TYX'},shape=(seg_list.shape))
    if args.method=="Deep Machine Learning" and args.superaccuracy=="False":
        im_list=np.array(im_list)
        if args.superaccuracy=="True":
            im_list=np.squeeze(im_list, axis=(3,))
        path=args.input_folder+'/Saved Segmentations/'+images+'_'+'original_padded.tif'
        tifffile.imwrite(path, im_list,imagej=True,metadata={'axes': 'TYX'},shape=(im_list.shape))
        del im_list
    del seg_list
    sleep(0.02)


print("I'm done")
print("it took me",time() - start_time, "seconds to do my job!")
print("Um9vdCByb2NrISB6bG9vcDI3")
#collect garbage files
gc.collect()
