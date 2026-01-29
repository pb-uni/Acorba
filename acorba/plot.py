def plot_imgs2(#for scanner
        org_imgs, folder,inc,
        pred_imgs=None,
        nm_img_to_plot=5,
        figsize=4,
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
        nm_img_to_plot, cols, figsize=(cols * figsize, nm_img_to_plot * figsize),squeeze=True)
    axes[0, 0].set_title("original", fontsize=15)
    axes[0, 1].set_title("prediction", fontsize=15)
    axes[0, 2].set_title("overlay", fontsize=15)
    for m in range(0, nm_img_to_plot):
        axes[m, 0].imshow(org_imgs[im_id], cmap=get_cmap(org_imgs))
        #axes[m, 0].axis('equal')
        axes[m, 0].set_axis_off()
        axes[m, 1].imshow(pred_imgs[im_id], cmap=get_cmap(pred_imgs))
        #axes[m, 1].axis('equal')
        axes[m, 1].set_axis_off()
        axes[m, 2].imshow(org_imgs[im_id], cmap=get_cmap(org_imgs))
        #axes[m, 2].axis('equal')
        axes[m, 2].imshow(
        mask_to_rgba(zero_pad_mask(pred_imgs[im_id], desired_size=org_imgs_size),
                    color=color,
                ),
        cmap=get_cmap(pred_imgs),
        alpha=alpha,
            )
        axes[m, 2].set_axis_off()
        im_id += 1
    plt.subplots_adjust(wspace=0)
    plot_name=folder+'/'+str(inc)+'_prediction'+kind+'.png'
    fig.savefig(plot_name, dpi=300, bbox_inches = 'tight',pad_inches = 0)
    plt.close(fig)
