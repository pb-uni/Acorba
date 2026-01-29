#Load libraries

import os
import cv2
import gc
from tensorflow.autograph import set_verbosity
from skimage.io import imread
import numpy as np
from patchify import unpatchify
import acorba

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
set_verbosity(0)

#retrieve user inputs from the GUI
args=acorba.folder.parser()

#Debug
#args=acorba.debug.scanner_debug()

#Set random seed to "The Answer to the Ultimate Question of Life, The Universe, and Everything."
np.random.seed(42)

#Importing machine learning libraries and models
if len(args.custom)>0:
    from keras.models import model_from_json
    print('Custom models and weights loading:')
    model_weights=os.listdir(args.custom)
    for i in model_weights:
        if i.endswith(".json"):
            with open(args.custom+"/"+i, 'r') as json_file:
                loaded_model_json = json_file.read()
            model = model_from_json(loaded_model_json)
        if i.endswith('.h5'):
            model.load_weights(args.custom+"/"+i)

else:
    from keras_unet.models import satellite_unet
    model = satellite_unet(input_shape=(256, 256, 1))
    model.load_weights("models/unetnocontrast_lessaugm_09092021_87.h5")
    model_accuracy = satellite_unet(input_shape=(256, 256, 1))
    model_accuracy.load_weights("models/unetnocontrast_lessaugm_09092021_22.h5")
print("Scanner prediction model and weights loaded")

#Retrieve image list and settting of various parameters
listimage_temp= os.listdir(args.input_folder)
listimage=[]
for i in listimage_temp:
    if i.endswith(".tif"):
        listimage.append(i)
listimage.sort()

IMG_WIDTH=256
IMG_HEIGHT=256
stack2=0
XML=[]
XT=[]
XML_accu=[]
XT2=[]
original=[]
color = [0]
for images in listimage:#loop for .tif stacks
    im=imread(args.input_folder+'/'+images)
    if min(im.shape)==im.shape[2]:#be sure the shape is not (height,width,timeframes) but(tf,height,width)
        im=np.rollaxis(im,2,0)
    im=im[len(im)-1] #use last timeframe instead of first as troubles come with bending not flat roots
    
    ####Machine learning not super accuracy####
    #Padding of the image to a integer number of 256x256 tiles and tiling it
    new_im, X, factorv, factorh=acorba.scanner.padding_tiling(image_input=im, tile=256)
    
    #Preprocessing of array for fitting the model data input
    X=np.array(X)
    X=X/255#Value between 0 and 1
    X=np.expand_dims(X,3)
    preds_train = model.predict(X, verbose=2,batch_size=1)
    preds_train_t = (preds_train > 0.5).astype(np.uint8)
    preds_train_t.shape=(factorv,factorh,256,256)#architecture of the patching
    mega=unpatchify(preds_train_t, new_im.shape)
    mega=cv2.resize(mega, (int(mega.shape[1] * 60 / 100),int(mega.shape[0] * 60 / 100)))
    
    ####Machine learning super accuracy####
    if args.superaccuracy=="True":
        im_expanded=np.expand_dims(new_im,2)#input is (256,256,1)
        im_expanded=im_expanded/255#data normalization between 0 and 1
        predictions_smooth = acorba.scanner.predict_img_with_smooth_windowing(
            input_img=im_expanded,
            window_size=256,
            subdivisions=2,  # Minimal amount of overlap for windowing. Must be an even number.
            nb_classes=1,
            pred_func=(
                lambda img_batch_subdiv: model_accuracy.predict((img_batch_subdiv), verbose=2,batch_size=1)
            )
        )
        mega_accu = (predictions_smooth > 0.01).astype(np.uint8)#somehow the non root material is under 0.01 instead of 0.5 for non smooth prediction
    else:
        mega_accu = np.zeros(shape=[int(mega.shape[1] * 60 / 100),int(mega.shape[0] * 60 / 100)], dtype=np.uint8)
    #####Trad entropy#####
    mega_t=acorba.scanner.trad_scanner(new_im)
    mega_t=cv2.resize(mega_t, (int(mega_t.shape[1] * 60 / 100),int(mega_t.shape[0] * 60 / 100)))
    
    ####Trad contrast threshold#####
    mega_t2=acorba.scanner.trad_scanner_threshold(new_im)
    mega_t2=cv2.resize(mega_t2, (int(mega_t2.shape[1] * 60 / 100),int(mega_t2.shape[0] * 60 / 100)))

    XML.append(mega)
    XML_accu.append(mega_accu)
    XT.append(mega_t)
    XT2.append(mega_t2)
    new_im=cv2.resize(new_im, (int(new_im.shape[1] * 60 / 100),int(new_im.shape[0] * 60 / 100)))
    original.append(new_im)
    del im
    print("image "+images+" done")

#max squared size
desired_size=1000

#padding every picture to a square
i=0
while i<len(XML):
    images=XML[i]
    old_size = images.shape # old_size is in (height, width) format
    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    img_t = cv2.resize(images, (new_size[1], new_size[0]))
    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    XML[i] = cv2.copyMakeBorder(img_t, top, bottom, left, right, cv2.BORDER_CONSTANT,value=0)
    i=i+1


i=0
while i<len(XML_accu):
    images=XML_accu[i]
    old_size = images.shape # old_size is in (height, width) format
    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    img_t = cv2.resize(images, (new_size[1], new_size[0]))
    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    XML_accu[i] = cv2.copyMakeBorder(img_t, top, bottom, left, right, cv2.BORDER_CONSTANT,value=0)
    i=i+1

i=0
while i<len(XT):
    images=XT[i]
    old_size = images.shape # old_size is in (height, width) format
    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    img_t = cv2.resize(images, (new_size[1], new_size[0]))
    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    XT[i] = cv2.copyMakeBorder(img_t, top, bottom, left, right, cv2.BORDER_CONSTANT,value=0)
    i=i+1

i=0
while i<len(XT2):
    images=XT2[i]
    old_size = images.shape # old_size is in (height, width) format
    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    img_t = cv2.resize(images, (new_size[1], new_size[0]))
    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    XT2[i] = cv2.copyMakeBorder(img_t, top, bottom, left, right, cv2.BORDER_CONSTANT,value=0)
    i=i+1
    
i=0
while i<len(original):
    images=original[i]
    old_size = images.shape # old_size is in (height, width) format
    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    img_t = cv2.resize(images, (new_size[1], new_size[0]))
    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    original[i] = cv2.copyMakeBorder(img_t, top, bottom, left, right, cv2.BORDER_CONSTANT,value=0)
    i=i+1
    
XML=np.array(XML)
XML_accu=np.array(XML_accu)
XT=np.array(XT)
XT2=np.array(XT2)
original=np.array(original)

acorba.scanner.test_segmentations2(org_imgs=original,
                    prediction=XML,
                    prediction2=XML_accu,
                    prediction_trad=XT,
                    prediction_trad2=XT2,
                    folder=args.input_folder,
                    listnames=listimage,
                    nm_img_to_plot=len(original),
                    accu=args.superaccuracy)

print("Segmentation test done!")
print("The segmentation test was saved in your experiment folder")
print("Um9vdCByb2NrISB6bG9vcDI3")

gc.collect()
