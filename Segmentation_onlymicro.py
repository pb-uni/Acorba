# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 17:08:31 2021
Modified on Jan 26 2026

Adds safe output directory handling (CLI + GUI fallback)

@author: Nelson / Modified by Philippe B.

"""


# Load libraries
print("Loading modules")
from time import time, sleep
start_time = time()
import numpy as np
import os
from pathlib import Path
import tifffile as tiff
from skimage.transform import resize
import cv2
from utils import trad_micro_sandwich, trad_micro_through, looplenght_determinator
import FreeSimpleGUI as sg
import argparse
import gc
import tifffile
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
from tensorflow.autograph import set_verbosity
set_verbosity(0)
print("ALL LIBRARIES ARE IMPORTED, ACORBA WILL START NOW")

# ----------------------
# Argument parsing
# ----------------------
parser = argparse.ArgumentParser(description='ACORBA')
parser.add_argument('--input_folder', type=str)
parser.add_argument('--exp_type', type=str)
parser.add_argument('--saveplot', type=str)
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
parser.add_argument('--output_dir', type=str, help='Folder where outputs are saved')
args = parser.parse_args()

print('user inputs')
rootfolder = args.input_folder
print('Root folder: ' + str(rootfolder))

# ----------------------
# Validate input folder
# ----------------------

if not rootfolder or not os.path.isdir(rootfolder):
    sg.theme('Light Brown 3')
    sg.popup(
        "Please select the input folder containing .tif files",
        title="Input folder required"
    )
    rootfolder = sg.popup_get_folder(
        "Select input folder",
        no_window=True
    )

if not rootfolder:
    raise SystemExit("No input folder selected. Exiting.")

rootfolder = os.path.abspath(os.path.expanduser(rootfolder))

saveplot = args.saveplot
rootplot = args.rootplot
save_segmentation = args.savesegmentation
exp_type = args.exp_type
normalization = args.normalization
predictiont = args.prediction
seg_method = args.method
DML_super_accuracy = args.superaccuracy
binary_folder = args.binary_folder
custom_models = args.custom
deactivate_smoothing = args.smooth
trad_method = args.tradmethod
print("Traditional method: " + str(trad_method))

# ----------------------
# Output directory logic
# ----------------------

def resolve_output_dir(rootfolder, user_output):
    if user_output:
        out = Path(user_output).expanduser()
        if not out.is_absolute() and rootfolder:
            out = Path(rootfolder).expanduser() / out
    else:
        if rootfolder:
            out = Path(rootfolder).expanduser() / 'Saved Segmentations'
        else:
            out = Path.home() / 'Saved Segmentations'
    return out


def ensure_writable_dir(primary: Path) -> Path:
    try:
        primary.mkdir(parents=True, exist_ok=True)
        testfile = primary / '.write_test'
        with open(testfile, 'w') as f:
            f.write('ok')
        testfile.unlink(missing_ok=True)
        return primary
    except Exception as e:
        print(f"Warning: cannot write to {primary} ({e}). Falling back to home directory.")
        fallback = Path.home() / 'ACORBA_Saved_Segmentations'
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback

output_dir = ensure_writable_dir(resolve_output_dir(rootfolder, args.output_dir))
print('Using output folder: ' + str(output_dir))

# ----------------------
# Processing parameters
# ----------------------
np.random.seed(42)
size = 256

# ----------------------
# Load DML models and weights
# ----------------------
if len(binary_folder) == 0:
    if len(custom_models) > 0:
        from keras.models import model_from_json
        print('Custom models and weights loading:')
        model_weights = os.listdir(custom_models)
        custom_models_list = []
        custom_weights_list = []
        for i in model_weights:
            if i.endswith('.json'):
                custom_models_list.append(i)
            if i.endswith('.h5'):
                custom_weights_list.append(i)
        ans_surface = 'surface'
        ans_tip = 'tip'
        for title in custom_models_list:
            if ans_surface in title:
                with open(Path(custom_models) / title, 'r') as json_file:
                    loaded_model = model_from_json(json_file.read())
                print('root surface model loaded')
            elif ans_tip in title:
                with open(Path(custom_models) / title, 'r') as json_file:
                    loaded_model_tip = model_from_json(json_file.read())
                print('root tip model loaded')
        for title in custom_weights_list:
            if ans_surface in title:
                loaded_model.load_weights(str(Path(custom_models) / title))
                print('root surface weights loaded')
            elif ans_tip in title:
                loaded_model_tip.load_weights(str(Path(custom_models) / title))
                print('root tip weights loaded')
    else:
        from keras_unet.models import satellite_unet
        print('Default models and weights loading')
        loaded_model = satellite_unet(input_shape=(size, size, 1))
        loaded_model_tip = satellite_unet(input_shape=(size, size, 1))
        if exp_type == 'Microscopy Through':
            loaded_model.load_weights('models/model_through_256_sat_jaccard_update16062021_54.h5')
            loaded_model_tip.load_weights('models/model_through_TIP2_256_sat_jaccard_update17062021_176.h5')
        else:
            loaded_model.load_weights('models/sandwichsurface_256_jaccard_update24062021_145.h5')
            loaded_model_tip.load_weights('models/microsandtip_256_sat_jaccard_update24062021_71.h5')
    print('Models and weights are loaded')

# ----------------------
# Input files
# ----------------------
rootlist = [i for i in os.listdir(rootfolder) if i.endswith('.tif')]
print('The following files will be processed')
print(rootlist)

looplenght = looplenght_determinator(list_image=rootlist, folder=rootfolder)

# ----------------------
# Main loop
# ----------------------
for inc_root in rootlist:
    sg.theme('Light Brown 8')
    print('Processing file ' + inc_root)
    test = tiff.imread(Path(rootfolder) / inc_root)
    tradlist = []
    X_test = np.zeros((len(test), size, size), dtype=np.float32)

    print('Converting stack to an array and resizing/padding to 256x256')
    for n in range(len(test)):
        sg.OneLineProgressMeter(inc_root + ' import, padding, resizing', n + 1, len(test), 'key')
        im = (test[n] / 255).astype('uint8')
        tradlist.append(im)
        if im.shape[0] != im.shape[1]:
            desired_size = size
            old_size = im.shape
            ratio = float(desired_size) / max(old_size)
            new_size = tuple([int(x * ratio) for x in old_size])
            img_t = cv2.resize(im, (new_size[1], new_size[0]))
            delta_w = desired_size - new_size[1]
            delta_h = desired_size - new_size[0]
            top, bottom = delta_h // 2, delta_h - (delta_h // 2)
            left, right = delta_w // 2, delta_w - (delta_w // 2)
            color = np.average(im)
            img_t = cv2.copyMakeBorder(img_t, top, bottom, left, right,
                                       cv2.BORDER_CONSTANT, value=color)
            X_test[n] = img_t / 255
        else:
            X_test[n] = resize(im, (size, size), mode='constant', preserve_range=True) / 255

    # Save resized stack
    tifffile.imwrite(str(output_dir / f"{inc_root}_original_resized256.tif"), X_test)
    tradlist = np.array(tradlist)

    # Predictions
    if seg_method == 'Deep Machine Learning':
        X_test_exp = np.expand_dims(X_test, 3)
        preds_test = loaded_model.predict(X_test_exp, verbose=2, batch_size=1)
        prediction = (preds_test > 0.5).astype(np.uint8)
        preds_test = loaded_model_tip.predict(X_test_exp, verbose=2, batch_size=1)
        prediction_tip = (preds_test > 0.5).astype(np.uint8)
    elif seg_method == 'Own masks':
        prediction = tiff.imread(Path(binary_folder) / 'surface' / inc_root)
        prediction_tip = tiff.imread(Path(binary_folder) / 'tip' / inc_root)
        prediction = resize(prediction, (len(prediction), size, size), mode='constant', preserve_range=True)
        prediction_tip = resize(prediction_tip, (len(prediction_tip), size, size), mode='constant', preserve_range=True)
        if np.max(prediction) > 1:
            prediction = prediction / np.max(prediction)
        if np.max(prediction_tip) > 1:
            prediction_tip = prediction_tip / np.max(prediction_tip)
    else:
        prediction = []
        if exp_type == 'Microscopy Through':
            for img_pred in X_test:
                prediction.append(trad_micro_through(img_pred.copy()))
        else:
            for img_pred in tradlist:
                prediction.append(trad_micro_sandwich(img_pred.copy()))
        prediction = np.array(prediction)
        X_test_exp = np.expand_dims(X_test, 3)
        preds_test = loaded_model_tip.predict(X_test_exp, verbose=2, batch_size=1)
        prediction_tip = (preds_test > 0.5).astype(np.uint8)

    tifffile.imwrite(str(output_dir / f"{inc_root}_root_tip_256.tif"), prediction_tip)
    tifffile.imwrite(str(output_dir / f"{inc_root}_root_surface_256.tif"), prediction)

    gc.collect()
    sleep(0.02)

print("I'm done")
print('it took me', time() - start_time, 'seconds to do my job!')