def fill_holes(binary_image):
    '''
Set of erode and dilate function from opencv to close holes
    '''
    import cv2
    import numpy as np
    kernel = np.ones((2,2),np.uint8)
    binary_image=cv2.dilate(binary_image,kernel,iterations=5)
    binary_image=cv2.erode(binary_image,kernel,iterations=5)
    binary_image=cv2.dilate(binary_image,kernel,iterations=2)
    binary_image=cv2.erode(binary_image,kernel,iterations=5)
    return binary_image

def binary_cleaner(image):
    import numpy as np
    from skimage import measure
    labels_mask = measure.label(image)                       
    regions = measure.regionprops(labels_mask)
    regions.sort(key=lambda x: x.area, reverse=True)
    if len(regions) > 1:
        for rg in regions[1:]:
            labels_mask[rg.coords[:,0], rg.coords[:,1]] = 0
        labels_mask[labels_mask!=0] = 1   
    return labels_mask.astype(np.uint8)
	
def plot_imgs(#for microscope
        org_imgs, folder,inc,
        pred_imgs=None,
        nm_img_to_plot=5,
        figsize=2,
        alpha=0.5,
        color="magenta",
        kind=""):
    from keras_unet.utils import reshape_arr, zero_pad_mask,get_cmap,mask_to_rgba
    import matplotlib.pyplot as plt
    if nm_img_to_plot > org_imgs.shape[0]:
        nm_img_to_plot = org_imgs.shape[0]
        
    im_id = 0
    org_imgs_size = org_imgs.shape[1]
    org_imgs = reshape_arr(org_imgs)
    cols = 3
    fig, axes = plt.subplots(
        nm_img_to_plot, cols, figsize=(cols * figsize, nm_img_to_plot * figsize), squeeze=True)
    axes[0, 0].set_title("original", fontsize=15)
    axes[0, 1].set_title("prediction", fontsize=15)
    axes[0, 2].set_title("overlay", fontsize=15)
    for m in range(0, nm_img_to_plot):
        axes[m, 0].imshow(org_imgs[im_id], cmap=get_cmap(org_imgs))
        axes[m, 0].set_axis_off()
        axes[m, 1].imshow(pred_imgs[im_id], cmap=get_cmap(pred_imgs))
        axes[m, 1].set_axis_off()
        axes[m, 2].imshow(org_imgs[im_id], cmap=get_cmap(org_imgs))
        axes[m, 2].imshow(
        mask_to_rgba(zero_pad_mask(pred_imgs[im_id], desired_size=org_imgs_size),
                    color=color,
                ),
        cmap=get_cmap(pred_imgs),
        alpha=alpha,
            )
        axes[m, 2].set_axis_off()
        im_id += 1
    plot_name=folder+'/'+str(inc)+'_prediction'+kind+'.png'
    plt.subplots_adjust(wspace=0)
    fig.savefig(plot_name, dpi=300, bbox_inches = 'tight',pad_inches = 0)
    plt.close(fig)
	
def trad_micro_through (img):
    from skimage.filters import threshold_otsu
    import numpy as np
    import cv2
    img2=img
    thresh = threshold_otsu(img2)
    binary = img2 <= thresh
    binary = np.multiply(binary, 1)
    binary= binary.astype('uint8') 
    kernel = np.ones((30,30),np.uint8)
    img2=cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return img2

def trad_micro_sandwich(img):
    from skimage.filters.rank import entropy
    from skimage.morphology import disk
    from skimage.transform import resize
    from skimage.filters import threshold_otsu
    from skimage import exposure, img_as_ubyte
    import numpy as np
    import cv2
    
    size=np.max(img.shape)
    if img.shape[0]!=img.shape[1]:
        old_size = img.shape # old_size is in (height, width) format
        ratio = float(size)/max(old_size)
        new_size = tuple([int(x*ratio) for x in old_size])
        img_t = cv2.resize(img, (new_size[1], new_size[0]))
        delta_w = size - new_size[1]
        delta_h = size - new_size[0]
        top, bottom = delta_h//2, delta_h-(delta_h//2)
        left, right = delta_w//2, delta_w-(delta_w//2)
        color = np.average(img)
        img_t = cv2.copyMakeBorder(img_t, top, bottom, left, right, cv2.BORDER_CONSTANT,value=color)
        img2=img_t
    else:
        img2 = resize(img, (size, size), mode='constant', preserve_range=True)
    
    img2 = img_as_ubyte(exposure.rescale_intensity(img2))
    entropy_img = entropy(img2, disk(4))
    entropy_img=img_as_ubyte(exposure.rescale_intensity(entropy_img))
    thresh = threshold_otsu(entropy_img)
    thresh1 = entropy_img <= thresh
    thresh1= np.multiply(thresh1, -1)
    thresh1=(cv2.bitwise_not(thresh1)*-1)
    thresh1=binary_cleaner(thresh1)
    kernel = np.ones((40,40),np.uint8)
    img2=cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel)
    img2 = resize(img2, (256, 256), mode='constant', preserve_range=True)#/255
    img2=img2.astype(np.uint8)
    return img2
	
def padding_resizing(image_input,size):
    '''
    Pad an image to a certain size with the average pixel value of the said image
    '''
    import numpy as np
    import cv2
    desired_size = size
    old_size = image_input.shape # old_size is in (height, width) format
    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    img_t = cv2.resize(image_input, (new_size[1], new_size[0]))
    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)
    color = np.average(image_input)
    img_t = cv2.copyMakeBorder(img_t, top, bottom, left, right, cv2.BORDER_CONSTANT,value=color)
    return img_t

def test_segmentations(
        org_imgs,
        prediction, prediction_tip,prediction_trad,folder,
        nm_img_to_plot,listnames,
        figsize=3,
        alpha=0.5,
        kind=""):
    from keras_unet.utils import reshape_arr, zero_pad_mask,get_cmap,mask_to_rgba
    import matplotlib.pyplot as plt
    im_id = 0
    org_imgs_size = org_imgs.shape[1]
    org_imgs = reshape_arr(org_imgs)
    cols = 7
    print( )
    if nm_img_to_plot==1:
        
        fig, axes = plt.subplots(
            nm_img_to_plot+1, cols, figsize=(cols * figsize, nm_img_to_plot+1 * figsize), squeeze=True)
    else:
        fig, axes = plt.subplots(
            nm_img_to_plot, cols, figsize=(cols * figsize, nm_img_to_plot * figsize), squeeze=True)
    axes[0, 0].set_title("original")
    axes[0, 1].set_title("prediction surface")
    axes[0, 2].set_title("overlay surface")
    axes[0,3].set_title("prediction tip")
    axes[0,4].set_title("overlay surface")
    axes[0,5].set_title("Traditional segmentation")
    axes[0,6].set_title("overlay traditionnal")
    for m in range(0, nm_img_to_plot):
        axes[m, 0].imshow(org_imgs[im_id], cmap=get_cmap(org_imgs))
        axes[m, 0].set_axis_off()
        axes[m, 0].set_title(listnames[m])
        axes[m, 1].imshow(prediction[im_id], cmap=get_cmap(prediction))
        axes[m, 1].set_axis_off()
        axes[m, 2].imshow(org_imgs[im_id], cmap=get_cmap(org_imgs))
        axes[m, 2].imshow(
        mask_to_rgba(zero_pad_mask(prediction[im_id], desired_size=org_imgs_size),
                    color="green",
                ),
        cmap=get_cmap(prediction),
        alpha=alpha,
            )
        axes[m, 2].set_axis_off()
        axes[m, 3].imshow(prediction_tip[im_id], cmap=get_cmap(prediction_tip))
        axes[m, 3].set_axis_off()
        axes[m, 4].imshow(org_imgs[im_id], cmap=get_cmap(org_imgs))
        axes[m, 4].imshow(
        mask_to_rgba(zero_pad_mask(prediction_tip[im_id], desired_size=org_imgs_size),
                    color='magenta',
                ),
        cmap=get_cmap(prediction_tip),
        alpha=alpha,
            )
        axes[m, 4].set_axis_off()
        axes[m, 5].imshow(prediction_trad[im_id], cmap=get_cmap(prediction_trad))
        axes[m, 5].set_axis_off()
        axes[m, 6].imshow(org_imgs[im_id], cmap=get_cmap(org_imgs))
        axes[m, 6].imshow(
        mask_to_rgba(zero_pad_mask(prediction_trad[im_id], desired_size=org_imgs_size),
                    color='red',
                ),
        cmap=get_cmap(prediction_trad),
        alpha=alpha,
            )
        axes[m, 6].set_axis_off()
        im_id += 1
    plot_name=folder+'/test_segmentation.png'
    fig.savefig(plot_name, dpi=300, bbox_inches = 'tight',pad_inches = 0)
    plt.show(block=False)
