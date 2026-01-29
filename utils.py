"""
Module containing all the function that ACORBA needs to run
"""

def subtract_background(image, radius=50, light_bg=False):
    from skimage.morphology import white_tophat, black_tophat, disk
    str_el = disk(radius) #you can also use 'ball' here to get a slightly smoother result at the cost of increased computing time
    if light_bg:
        return black_tophat(image, str_el)
    else:
        return white_tophat(image, str_el)

def find_all(a_str, sub):
                start = 0
                while True:
                    start = a_str.find(sub, start)
                    if start == -1: return
                    yield start
                    start += len(sub) # use start += 1 to find overlapping matches


def parser():
    '''
Import all the GUI user define information through the console parser
    '''
    import argparse
    parser = argparse.ArgumentParser(description='ACORBA')
    parser.add_argument('--input_folder', type=str)
    parser.add_argument('--exp_type', type=str)
    parser.add_argument('--normalization', type=str)
    parser.add_argument('--prediction', type=str)
    parser.add_argument('--binary_folder', type=str)
    parser.add_argument('--rootplot', type=str)
    parser.add_argument('--method', type=str)
    parser.add_argument('--custom', type=str)
    parser.add_argument('--smooth', type=str)
    parser.add_argument('--superaccuracy', type=str)
    parser.add_argument('--savesegmentation', type=str)
    parser.add_argument('--tradmethod', type=str)
    parser.add_argument("--save_lenghts", type=str)
    parser.add_argument("--scale", type=str)
    args = parser.parse_args()
    print('User inputs')
    print('Root folder: '+args.input_folder)
    print('Save root plot: '+args.rootplot)
    print('Save raw segmentation: '+args.savesegmentation)
    print('Experiment type: '+args.exp_type)
    print('Normalization: '+args.normalization)
    print('Save prediction plot: '+args.prediction)
    print('Segmentation method: '+args.method)
    print('Super accuracy mode for scanner: '+args.superaccuracy)
    print('Use masks from: '+args.binary_folder)
    print('Use models from: '+args.custom)
    print('Deactivate smoothing: '+args.smooth)
    print("Traditional method: "+args.tradmethod)
    print("Save lenghts: "+args.save_lenghts)
    print("Scale pixel per microns: "+args.scale)
    return args


    

















    



    
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



