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
    #parser.add_argument("--save_lenghts", type=str)
    #parser.add_argument("--scale", type=str)
    parser.add_argument("--circlepix", type=str)
    parser.add_argument("--broken", type=str)
    parser.add_argument("--vector", type=str)
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
    #print("Save lenghts: "+args.save_lenghts)
    #print("Scale pixel per microns: "+args.scale)
    print("Size cropping circle: "+args.circlepix)
    print("Broken skeleton factor: "+args.broken)
    print("Vector argument: "+args.vector)
    return args

def list_tif(folder):
    import os
    listimage_temp= os.listdir(folder)
    listimage = [i for i in listimage_temp if i.endswith(".tif")]
    listimage.sort()
    print("The following image will be processed:")
    print(listimage)
    return listimage

def looplenght_determinator(list_image,folder):
    import tifffile
    looplenght=0
    for i in list_image:
        test = tifffile.TiffFile(folder+'/'+i)
        looplenght=looplenght+int(len(test.pages))
        test.close()
    return looplenght
