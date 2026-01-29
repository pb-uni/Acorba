import numpy as np
from patchify import unpatchify
from fil_finder import FilFinder2D
import astropy.units as u
import warnings
from skimage.morphology import skeletonize
import cv2

def check_broken_skeleton(listends,listorigins,brokenfactor):
    yyo=0
    d_piece=[]
    d_length=[]
    while yyo<len(listends):
        yyy=0
        while yyy<len(listorigins):
            seg1=yyo
            seg2=yyy
            if seg1==seg2:
                d=10000
                d_piece.append([seg1,seg2])
                d_length.append(d)
            else:
                d=np.sqrt((listorigins[yyy][0] - listends[yyo][0]) ** 2 + (listorigins[yyy][1] - listends[yyo][1]) ** 2)
                d_piece.append([seg1,seg2])
                d_length.append(d)
            yyy=yyy+1
        yyo=yyo+1
    d_length_ar=np.array(d_length)
    mini=np.where(d_length_ar<brokenfactor)
    mini=mini[0]
    return mini,d_piece

def organize_pixels(coordinates_input,skel_img_input):
    '''
    Organize pixels of a strings into a closest to closest pixel list from left to right
    
    '''
    kernel=skel_img_input.copy()
    newskelx=[]
    newskely=[]
    loopit=0
    while loopit<len(np.where(skel_img_input==1)[0])-1:
        if loopit==0:
            col=coordinates_input[0]
            row=coordinates_input[1]
            newskelx.append(coordinates_input[1])
            newskely.append(coordinates_input[0])
        else:
            row=nextpoint[0]
            col=nextpoint[1]
        listneighbors=[kernel[row-1,col],
                       kernel[row+1,col],
                       kernel[row,col-1],
                       kernel[row,col+1],
                       kernel[row+1,col-1],
                       kernel[row+1,col+1],
                       kernel[row-1,col-1],
                       kernel[row-1,col+1]]
        listneighbors=np.array(listneighbors)
        listneighbors_skel=np.where(listneighbors==1)[0][0]
        if listneighbors_skel==0:
            nextpoint=[row-1,col]
        elif listneighbors_skel==1:
            nextpoint=[row+1,col]
        elif listneighbors_skel==2:
            nextpoint=[row,col-1]
        elif listneighbors_skel==3:
            nextpoint=[row,col+1]
        elif listneighbors_skel==4:
            nextpoint=[row+1,col-1]
        elif listneighbors_skel==5:
            nextpoint=[row+1,col+1]
        elif listneighbors_skel==6:
            nextpoint=[row-1,col-1]
        elif listneighbors_skel==7:
            nextpoint=[row-1,col+1]
            
        #draw start pixel as background
        kernel[row,col]=0
        newskelx.append(nextpoint[0])
        newskely.append(nextpoint[1])
        loopit=loopit+1
    return newskelx, newskely

def skeleton_endpoints(skel):
    '''
    Finds the endpoints of a skeleton or string of pixels
    '''
    import numpy as np
    import cv2
    # make out input nice, possibly necessary
    skel = skel.copy()
    skel[skel!=0] = 1
    skel = np.uint8(skel)
    # apply the convolution
    kernel = np.uint8([[1,  1, 1],
                       [1, 10, 1],
                       [1,  1, 1]])
    src_depth = -1
    filtered = cv2.filter2D(skel,src_depth,kernel)
    out = np.zeros_like(skel)
    out=np.where(filtered==11)
    return out

def angle_to(p1, p2, rotation=0, clockwise=True):
    '''
    Calculate the angle of a vector with two sets of coordinate
    '''
    from math import  atan2, degrees
    angle = degrees(atan2(p2[1] - p1[1], p2[0] - p1[0])) - rotation
    if not clockwise:
        angle = -angle
    return angle % 360

def root_angle(skeleton,originvector):
    #Determine skeleton endpoints with a kernel filter

    skelends=skeleton_endpoints(skeleton)
    
    #if several skeleton origins: take the average coordinates
    if len(skelends[1])>2:
        xc=0
        dc=[]
        while xc<len(skelends[1])-1:
            dc.append(np.sqrt((skelends[1][xc]-skelends[1][xc+1])**2))
            xc=xc+1
        dc=np.array(dc)
        mindc=int(np.where(dc==np.min(dc))[0])
        skelendslist=np.array(skelends)
        skelendslist=(np.delete(skelendslist[0], mindc),np.delete(skelendslist[1],mindc))
        skelends=tuple(skelendslist)

    listorigins=[]
    listends=[]
    #determine origins and ends list by their x coordinates
    p1temp=[skelends[1][0],skelends[0][0]]
    p2temp=[skelends[1][1],skelends[0][1]]
    if p1temp[0]>p2temp[0]:
        listorigins=p2temp
        listends=p1temp
    else:
        listorigins=p1temp
        listends=p2temp
        
    #Reorder skeleton by closest to closest coordinates
    newskely,newskelx=organize_pixels(coordinates_input=listends,skel_img_input=skeleton)
    
    #Find the origin of the vector (10 coordinates up the skeleton from the end)
    endminus=(newskelx[originvector],newskely[originvector])
    start=(listends[0],listends[1])

    anglee=angle_to(endminus,start)
    if anglee >180:
        anglee=anglee-360
    return anglee, listorigins, listends,newskely,newskelx,start,endminus


def root_skeleton(mega,areatresh):
    #Contours detection and tresholding
    contours, hierarchy = cv2.findContours(mega.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contour_list=[cnt for cnt in contours if cv2.contourArea(cnt)>areatresh and cv2.arcLength(cnt,True)>200]
    
    #Create an empty image to put the contours filled in
    empty=np.zeros([mega.shape[0],mega.shape[1]],dtype=np.uint8)
    
    #List of segmentation contours
    listseg=[cv2.drawContours(empty.copy(), contour_list,  cnt, (1), cv2.FILLED) for cnt in range(0,len(contour_list))]
    
    #Skeletonize and prune individually in cropped images corresponding to the bounding boxes of each roots
    listcroppedboxes=[]
    listxy=[]
    for (seg,contour) in zip(listseg,contour_list):
        x,y,w,h = cv2.boundingRect(contour)
        img_temp=seg
        add_rm=10
        if y-add_rm or x-add_rm<=0:
            ad_rm=5
            if y-add_rm or x-add_rm<=0:
                ad_rm=3
                if y-add_rm or x-add_rm<=0:
                    ad_rm=1
        listcroppedboxes.append(img_temp[y-add_rm:y+add_rm+h, x-ad_rm:x+ad_rm+w])
        listxy.append([x-ad_rm,y-add_rm])
    listskel=[skeletonize_prune(image_input=seg).skeleton_longpath for seg in listcroppedboxes]
    #listskel=[skeletonize_prune(image_input=seg).skeleton_longpath for seg in listseg]
    skeletons_detected=str(len(listskel))
    print(skeletons_detected,"Skeletons detected")
    return listskel,listxy

def skeletonize_prune(image_input):
    skeleton=skeletonize(image_input,method="lee")
    warnings.filterwarnings('ignore')
    fil = FilFinder2D(skeleton, distance=250 * u.pc, mask=skeleton)
    fil.medskel(verbose=False)
    fil.analyze_skeletons(branch_thresh=4* u.pix, skel_thresh=4 * u.pix, prune_criteria='length')
    warnings.filterwarnings('default')
    return fil

def trad_scanner(img):
    import numpy as np
    import cv2
    from skimage.filters.rank import entropy
    from skimage.morphology import disk
    entropy_img = entropy(img, disk(4))#10
    entropy_img=(entropy_img).astype('uint8')
    ret,binary = cv2.threshold(entropy_img,np.max(entropy_img)-2,np.max(entropy_img),cv2.THRESH_BINARY)
    kernel = np.ones((4,4),np.uint8)
    binary2=cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary2=cv2.morphologyEx(binary2, cv2.MORPH_OPEN, kernel)
    img2=binary2
    img2[img2==0]=0
    img2[img2>0]=1
    return img2
    
def trad_scanner_threshold(img):
    from skimage import exposure
    from skimage.filters import threshold_otsu
    import numpy as np
    import cv2
    from scipy import ndimage
    img[img==0]=np.mean(img[img>0])
    v_min, v_max = np.percentile(img, (0.2, 99.8))
    img = exposure.rescale_intensity(img, in_range=(v_min, v_max))
    img = ndimage.gaussian_filter(img, 3)
    thresh = threshold_otsu(img)
    binary = img <= thresh
    binary = np.multiply(binary, 1)
    binary= binary.astype('uint8')
    kernel = np.ones((10,10),np.uint8)#◙30
    binary2=cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    kernel = np.ones((1,2),np.uint8)#♣"35"
    binary2=cv2.morphologyEx(binary2, cv2.MORPH_CLOSE, kernel)
    img2 = cv2.bitwise_not(binary2)
    img2[img2==255]=1
    img2[img2==254]=0
    return img2
def padding_tiling(image_input, tile=256):
    """
    Pads an image to a certain size with black pixel_value=0
    The size is determined as the maximum number of tiles possible +1extra
    in h and w
    then tile this image to the tile size specified

    Parameters
    ----------
    image_input : np.array 8 bits
    tile : size of the tiles w and h. The default is 256. OPTIONAL

    Returns
    -------
    new_im : padded picture
    X : tile list
    factorv : how many tiles in width
    factorh : how many tiles in height

    """
    import cv2
    y1=0
    x1=0
    yd=tile
    xd=tile
    old_size = image_input.shape[:2]
    factorv=int(old_size[0]/tile)+1
    factorh=int(old_size[1]/tile)+1
    desired_sizev=(factorv*tile)
    desired_sizeh=(factorh*tile)
    delta_v = desired_sizev - old_size[0]
    delta_h = desired_sizeh - old_size[1]
    top, bottom = delta_v//2, delta_v-(delta_v//2)
    left, right = delta_h//2, delta_h-(delta_h//2)
    new_im = cv2.copyMakeBorder(image_input, top, bottom, left, right, cv2.BORDER_CONSTANT,value=0)#Actual padding
    X = [new_im[x1:x1+xd,y1:y1+yd] for x1 in range(0,new_im.shape[0],xd) for y1 in range(0,new_im.shape[1],yd)]
    return new_im, X, factorv, factorh


def DML_default(image,model):
    """
    takes a 8 bits scanner image, tile it to x*256*256 and return segmentation 
    using the default acorba scanner model
    
    Parameters
    ----------
    image : np.array of the 8 bits scanner image (1timeframe) to predict.
    model : keras DML model

    Returns
    -------
    mega : DML segmentation of image unpatchified
    new_im=from padding_tiling >padded image

    """
    
    #Padding of the image to a integer number of 256x256 tiles and tiling it
    new_im, X, factorv, factorh=padding_tiling(image_input=image, tile=256)

    #Preprocessing of array for fitting the model data input
    X=np.array(X)#transform X from a list to an array
    X=X/255#Value between 0 and 1: data normalization
    X=np.expand_dims(X,3)#Model input is (256,256,1)

    #Predictions
    preds_train = model.predict(X, verbose=2,batch_size=1)
    preds_train_t = (preds_train > 0.5).astype(np.uint8)

    #Reconstruction of a segmentation image from the tiles
    preds_train_t.shape=(factorv,factorh,256,256)#architecture of the patching
    mega=unpatchify(preds_train_t, new_im.shape)
    return mega,new_im

def DML_default_superaccu(image,model):
    im_expanded=np.expand_dims(image,2)#input is (256,256,1)
    im_expanded=im_expanded/255#data normalization between 0 and 1
    predictions_smooth = predict_img_with_smooth_windowing(
        input_img=im_expanded,
        window_size=256,
        subdivisions=2,  # Minimal amount of overlap for windowing. Must be an even number.
        nb_classes=1,
        pred_func=(
            lambda img_batch_subdiv: model.predict((img_batch_subdiv), verbose=2,batch_size=1)))
    mega = (predictions_smooth > 0.01).astype(np.uint8)
    return mega

import scipy.signal
from tqdm import tqdm
import gc
def _spline_window(window_size, power=2):
    intersection = int(window_size/4)
    wind_outer = (abs(2*(scipy.signal.triang(window_size))) ** power)/2
    wind_outer[intersection:-intersection] = 0
    wind_inner = 1 - (abs(2*(scipy.signal.triang(window_size) - 1)) ** power)/2
    wind_inner[:intersection] = 0
    wind_inner[-intersection:] = 0
    wind = wind_inner + wind_outer
    wind = wind / np.average(wind)
    return wind


cached_2d_windows = dict()
def _window_2D(window_size, power=2):
    # Memoization
    global cached_2d_windows
    key = "{}_{}".format(window_size, power)
    if key in cached_2d_windows:
        wind = cached_2d_windows[key]
    else:
        wind = _spline_window(window_size, power)
        wind = np.expand_dims(np.expand_dims(wind, 1), 1)      #SREENI: Changed from 3, 3, to 1, 1 
        wind = wind * wind.transpose(1, 0, 2)
        cached_2d_windows[key] = wind
    return wind


def _pad_img(img, window_size, subdivisions):
    aug = int(round(window_size * (1 - 1.0/subdivisions)))
    more_borders = ((aug, aug), (aug, aug), (0, 0))
    ret = np.pad(img, pad_width=more_borders, mode='reflect')
    return ret


def _unpad_img(padded_img, window_size, subdivisions):
    aug = int(round(window_size * (1 - 1.0/subdivisions)))
    ret = padded_img[
        aug:-aug,
        aug:-aug,
        :
    ]
    return ret


def _rotate_mirror_do(im):
    mirrs = []
    mirrs.append(np.array(im))
    mirrs.append(np.rot90(np.array(im), axes=(0, 1), k=1))
    mirrs.append(np.rot90(np.array(im), axes=(0, 1), k=2))
    mirrs.append(np.rot90(np.array(im), axes=(0, 1), k=3))
    im = np.array(im)[:, ::-1]
    mirrs.append(np.array(im))
    mirrs.append(np.rot90(np.array(im), axes=(0, 1), k=1))
    mirrs.append(np.rot90(np.array(im), axes=(0, 1), k=2))
    mirrs.append(np.rot90(np.array(im), axes=(0, 1), k=3))
    return mirrs


def _rotate_mirror_undo(im_mirrs):
    origs = []
    origs.append(np.array(im_mirrs[0]))
    origs.append(np.rot90(np.array(im_mirrs[1]), axes=(0, 1), k=3))
    origs.append(np.rot90(np.array(im_mirrs[2]), axes=(0, 1), k=2))
    origs.append(np.rot90(np.array(im_mirrs[3]), axes=(0, 1), k=1))
    origs.append(np.array(im_mirrs[4])[:, ::-1])
    origs.append(np.rot90(np.array(im_mirrs[5]), axes=(0, 1), k=3)[:, ::-1])
    origs.append(np.rot90(np.array(im_mirrs[6]), axes=(0, 1), k=2)[:, ::-1])
    origs.append(np.rot90(np.array(im_mirrs[7]), axes=(0, 1), k=1)[:, ::-1])
    return np.mean(origs, axis=0)


def _windowed_subdivs(padded_img, window_size, subdivisions, nb_classes, pred_func):
    WINDOW_SPLINE_2D = _window_2D(window_size=window_size, power=2)
    step = int(window_size/subdivisions)
    padx_len = padded_img.shape[0]
    pady_len = padded_img.shape[1]
    subdivs = []
    for i in range(0, padx_len-window_size+1, step):
        subdivs.append([])
        for j in range(0, pady_len-window_size+1, step):            
            patch = padded_img[i:i+window_size, j:j+window_size, :]
            subdivs[-1].append(patch)
    gc.collect()
    subdivs = np.array(subdivs)
    gc.collect()
    a, b, c, d, e = subdivs.shape
    subdivs = subdivs.reshape(a * b, c, d, e)
    gc.collect()
    subdivs = pred_func(subdivs)
    gc.collect()
    subdivs = np.array([patch * WINDOW_SPLINE_2D for patch in subdivs])
    gc.collect()
    # Such 5D array:
    subdivs = subdivs.reshape(a, b, c, d, nb_classes)
    gc.collect()
    return subdivs


def _recreate_from_subdivs(subdivs, window_size, subdivisions, padded_out_shape):
    step = int(window_size/subdivisions)
    padx_len = padded_out_shape[0]
    pady_len = padded_out_shape[1]
    y = np.zeros(padded_out_shape)
    a = 0
    for i in range(0, padx_len-window_size+1, step):
        b = 0
        for j in range(0, pady_len-window_size+1, step):
            windowed_patch = subdivs[a, b]
            y[i:i+window_size, j:j+window_size] = y[i:i+window_size, j:j+window_size] + windowed_patch
            b += 1
        a += 1
    return y / (subdivisions ** 2)


def predict_img_with_smooth_windowing(input_img, window_size, subdivisions, nb_classes, pred_func):
    pad = _pad_img(input_img, window_size, subdivisions)
    pads = _rotate_mirror_do(pad)
    res = []
    for pad in tqdm(pads):
        # For every rotation:
        sd = _windowed_subdivs(pad, window_size, subdivisions, nb_classes, pred_func)
        one_padded_result = _recreate_from_subdivs(
            sd, window_size, subdivisions,
            padded_out_shape=list(pad.shape[:-1])+[nb_classes])
        res.append(one_padded_result)
    # Merge after rotations:
    padded_results = _rotate_mirror_undo(res)
    prd = _unpad_img(padded_results, window_size, subdivisions)
    prd = prd[:input_img.shape[0], :input_img.shape[1], :]
    return prd

def test_segmentations2(
        org_imgs,
        prediction,prediction2,prediction_trad,prediction_trad2,folder,listnames,
        nm_img_to_plot,
        accu,
        figsize=30,
        alpha=0.5,
        kind=""):
    from keras_unet.utils import reshape_arr, zero_pad_mask,get_cmap,mask_to_rgba
    import matplotlib.pyplot as plt
    plt.rc('font', size='100')
    im_id = 0
    org_imgs_size = org_imgs.shape[1]
    org_imgs = reshape_arr(org_imgs)
    cols = 5
    fig, axes = plt.subplots(
        nm_img_to_plot, cols, figsize=(cols * figsize, nm_img_to_plot * figsize), squeeze=False)
    axes[0, 0].set_title("original")
    axes[0, 1].set_title("Machine learning")
    if accu=="True":
        axes[0, 2].set_title("Machine learning_accuracy")
    else:
        axes[0, 2].set_title("Mode not selected")
    axes[0, 3].set_title("Traditional_entropy")
    axes[0, 4].set_title("Traditional_threshold")
    for m in range(0, nm_img_to_plot):
        axes[m, 0].imshow(org_imgs[im_id], cmap=get_cmap(org_imgs))
        axes[m, 0].set_axis_off()
        axes[m, 0].set_title(listnames[m])
        axes[m, 1].imshow(org_imgs[im_id], cmap=get_cmap(org_imgs))
        axes[m, 1].imshow(
        mask_to_rgba(zero_pad_mask(prediction[im_id], desired_size=org_imgs_size),
                    color="green",
                ),
        cmap=get_cmap(prediction),
        alpha=alpha,
            )
        axes[m, 1].set_axis_off()
        axes[m, 2].imshow(org_imgs[im_id], cmap=get_cmap(org_imgs))
        axes[m, 2].imshow(
        mask_to_rgba(zero_pad_mask(prediction2[im_id], desired_size=org_imgs_size),
                    color="green",
                ),
        cmap=get_cmap(prediction2),
        alpha=alpha,
            )
        axes[m, 2].set_axis_off()
        axes[m, 3].imshow(org_imgs[im_id], cmap=get_cmap(org_imgs))
        axes[m, 3].imshow(
        mask_to_rgba(zero_pad_mask(prediction_trad[im_id], desired_size=org_imgs_size),
                    color='red',
                ),
        cmap=get_cmap(prediction_trad),
        alpha=alpha,
            )
        axes[m, 3].set_axis_off()
        axes[m, 4].imshow(org_imgs[im_id], cmap=get_cmap(org_imgs))
        axes[m, 4].imshow(
        mask_to_rgba(zero_pad_mask(prediction_trad2[im_id], desired_size=org_imgs_size),
                    color='red',
                ),
        cmap=get_cmap(prediction_trad2),
        alpha=alpha,
            )
        axes[m, 4].set_axis_off()
        im_id += 1
    plot_name=folder+'/test_segmentation.png'
    plt.savefig(plot_name)
    plt.show(block=False)
