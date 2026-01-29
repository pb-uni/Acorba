# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 22:09:48 2021

@author: Nelson
"""
import os
from tifffile import tifffile
import cv2
import numpy as np
from skimage.filters import threshold_otsu
from skimage.io import imread
import napari
import numpy
from napari.types import ImageData, LabelsData
import skimage
from magicgui import magicgui
from skimage.transform import resize
import acorba
import warnings
from napari.settings import get_settings
import PySimpleGUI as sg

args=acorba.folder.parser()
if args.exp_type=="Scanner":
    sg.popup('The Binary Assistant is not configured for scanner images yet, sorry',
             keep_on_top=True,title="Error")
    exit()
get_settings().application.ipy_interactive = True
i=0
main_dir=args.input_folder
listimage_temp=os.listdir(main_dir)
rootlist=[i for i in listimage_temp if i.endswith(".tif")]
image=imread(main_dir+'/'+rootlist[i],0)
image=resize(image, (image.shape[0],256, 256),anti_aliasing=True)
#image=image[0:5]
image2=np.asarray([acorba.utils.subtract_background(image=i, radius=15, light_bg=True) for i in image])
if not os.path.exists(main_dir+"/Segmentations/"):
    os.mkdir(main_dir+"/Segmentations/")
if not os.path.exists(main_dir+"/Segmentations/Surface/"):
        os.mkdir(main_dir+"/Segmentations/Surface/")
if not os.path.exists(main_dir+"/Segmentations/Tip/"):
        os.mkdir(main_dir+"/Segmentations/Tip/")
viewer = napari.Viewer()
viewer.add_image(image2, name=rootlist[i])

#PREVIOUS ROOT
@magicgui(call_button="Next root")
def next_root(viewer : napari.Viewer ):
    global i
    imagename=rootlist[i]
    tifffile.imwrite(main_dir+"/Segmentations//Surface/"+imagename, viewer.layers[1].data)
    tifffile.imwrite(main_dir+"/Segmentations//Tip/"+imagename,viewer.layers[3].data)
    i=i+1
    global image
    global image2
    if i >=len(rootlist):
        warnings.warn("No more root, close the viewer")
    else:
        imagename=rootlist[i]
        image=imread(args.input_folder+'/'+imagename,0)
        image=resize(image, (image.shape[0],256, 256),anti_aliasing=True)
        #image=image[0:5]
        image2=np.asarray([acorba.utils.subtract_background(image=i, radius=15, light_bg=True) for i in image])
        viewer.layers[0].data=image2
        viewer.layers[0].name=imagename
        viewer.layers[2].name="Rectangular selections"
        del viewer.layers['Root_Tip']
        del viewer.layers["Rectangular selections"]
        viewer.layers[0].visible=1
    return i, image2
#######################################"if i ==len(image) Stop napari
@magicgui(call_button="Create tip segmentation")
def tip_segmentation(layer: napari.layers.Labels, shapes_layer: napari.layers.Shapes, viewer : napari.Viewer):
    surface_seg=layer.data
    global image2
    if shapes_layer==None:
        shapes_layer = viewer.add_shapes(ndim=3)
        shapes_layer.mode = 'add_rectangle'
        warnings.warn("Please add rectangle selection on root tip")
        return
    if len(shapes_layer.data)==0:
        warnings.warn("Please add rectangle selection on root tip")
        return
    if len(shapes_layer.data) != len(surface_seg):
        warnings.warn("Check if there is not one rectangle missing or too much")
        return
    rectangles=shapes_layer.data
    #reorganize the rectangular selections list from 0 to x timeframes
    rectangles=np.asarray(rectangles)
    listc=[int(i[0][0]) for i in rectangles]
    newarray=np.zeros(rectangles.shape)
    i=0
    while i<len(rectangles):
        newarray[i]=rectangles[listc.index(i)]
        i+=1
        
    tip_seg=[]
    for i, j in zip(newarray, surface_seg):
        blank=np.zeros(j.shape)
        rec=cv2.rectangle(blank.copy(),pt1=(int(i[0][2]),int(i[0][1])),pt2=(int(i[2][2]),int(i[2][1])),color=1,thickness=-1)
        selection=rec*j
        tip_seg.append(selection.astype(int))
    tip_seg=np.asarray(tip_seg).astype(int)
    viewer.add_labels(tip_seg, name="Root_Tip") 
    viewer.layers[0].visible=0
    viewer.layers[2].visible=0
    viewer.layers[1].color={1: 'green'}
    viewer.layers[3].color={1: 'magenta'}

@magicgui(
    auto_call=True,
    Threshold={"widget_type": "FloatSlider", "max": np.max(image2)},
    close={"widget_type": "FloatSlider", "max": 10,"min":1},
    fill_holes={"widget_type": "Checkbox"},
    Dilate={"widget_type": "Checkbox"},
    layout='vertical')
def Root_Surface(layer: ImageData,Threshold: int = threshold_otsu(image2) ,
                 close: int = 5, fill_holes=True,Dilate=True)->LabelsData:
    if layer is not None:
        a=layer <= Threshold
        a=1-(a*1)
        from skimage.morphology import square
        from scipy.ndimage import binary_fill_holes
        sq=close
        if fill_holes:   
            b=np.asarray([skimage.morphology.closing(i,square(int(sq))) for i in a])
            c=np.asarray([binary_fill_holes(i) for i in b])
        else:
            c=layer
        d=np.asarray([acorba.microscope.binary_cleaner(i) for i in c]) 
        if Dilate:
            e=np.asarray([skimage.morphology.binary_dilation(i) for i in d]) 
        else:
            e=d
        return e
@viewer.bind_key('d')
def next_step(viewer):
    current_step=viewer.dims.current_step[0]
    viewer.dims.set_current_step(0,current_step+1)
@viewer.bind_key('q')
def previous_step(viewer):
    current_step=viewer.dims.current_step[0]
    viewer.dims.set_current_step(0,current_step-1)
viewer.window.add_dock_widget(Root_Surface, area="bottom",name="Thresholding")
viewer.window.add_dock_widget(tip_segmentation, area="bottom")
viewer.window.add_dock_widget(next_root,area='bottom')
#next_root.called.connect(Root_Surface)
#viewer.layers.events.inserted.connect(Root_Surface.reset_choices)
#viewer.layers.events.removed.connect(Root_Surface.reset_choices)
napari.run()

#add progress bar
#add key binding for changing slide