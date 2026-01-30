#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ACORBA - Segmentation Only (Microscopy)
Performs microscopy root segmentation only with no angle measurements.

Created on Mon Sep 13 17:08:31 2021
Modified on Jan 26 2026 by Philippe Baumann

Improvements:
- Safe output directory handling with CLI + GUI fallback
- Cross-platform compatibility (macOS/Linux/Windows)
- Better error handling and user feedback
- FreeSimpleGUI instead of PySimpleGUI
- Proper Path handling with pathlib

@author: Nelson BC Serre / Modified by Philippe B.
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
import FreeSimpleGUI as sg
import argparse
import gc

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
from tensorflow.autograph import set_verbosity
set_verbosity(0)

print("ALL LIBRARIES ARE IMPORTED, ACORBA WILL START NOW")
print("=" * 80)

# ----------------------
# Utility Functions
# ----------------------

def looplenght_determinator(list_image, folder):
    """
    Calculate total number of frames across all image stacks.
    Used for progress bar calculations.
    """
    total_frames = 0
    for img_name in list_image:
        img_path = Path(folder) / img_name
        try:
            img = tiff.imread(str(img_path))
            total_frames += len(img)
        except Exception as e:
            print(f"Warning: Could not read {img_name} for length calculation: {e}")
    return total_frames


def trad_micro_through(image):
    """
    Traditional segmentation method for microscopy through experiments.
    Uses entropy-based thresholding.
    """
    from skimage.filters import threshold_otsu
    try:
        thresh = threshold_otsu(image)
        binary = image > thresh
        return binary.astype(np.uint8)
    except Exception as e:
        print(f"Warning: Traditional segmentation failed, returning zeros: {e}")
        return np.zeros_like(image, dtype=np.uint8)


def trad_micro_sandwich(image):
    """
    Traditional segmentation method for microscopy sandwich experiments.
    Uses adaptive thresholding.
    """
    try:
        # Convert to uint8 if needed
        if image.dtype != np.uint8:
            image_8bit = (image * 255).astype(np.uint8)
        else:
            image_8bit = image
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            image_8bit, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )
        return (binary > 0).astype(np.uint8)
    except Exception as e:
        print(f"Warning: Traditional segmentation failed, returning zeros: {e}")
        return np.zeros_like(image, dtype=np.uint8)


# ----------------------
# Argument parsing
# ----------------------
parser = argparse.ArgumentParser(description='ACORBA - Segmentation Only (Microscopy)')
parser.add_argument('--input_folder', type=str, help='Folder containing .tif stacks')
parser.add_argument('--exp_type', type=str, help='Experiment type')
parser.add_argument('--saveplot', type=str, default='0')
parser.add_argument('--normalization', type=str, default='1')
parser.add_argument('--prediction', type=str, default='None')
parser.add_argument('--binary_folder', type=str, default='')
parser.add_argument('--rootplot', type=str, default='0')
parser.add_argument('--method', type=str, default='Deep Machine Learning')
parser.add_argument('--custom', type=str, default='')
parser.add_argument('--smooth', type=str, default='1')
parser.add_argument('--superaccuracy', type=str, default='0')
parser.add_argument('--savesegmentation', type=str, default='0')
parser.add_argument('--tradmethod', type=str, default='Entropy')
parser.add_argument('--circlepix', type=str, default='40')
parser.add_argument('--output_dir', type=str, help='Folder where outputs are saved')
args = parser.parse_args()

print('User inputs:')
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
print(f'Using input folder: {rootfolder}')

# Parse other arguments
exp_type = args.exp_type
seg_method = args.method
binary_folder = args.binary_folder
custom_models = args.custom
trad_method = args.tradmethod

print(f"Experiment type: {exp_type}")
print(f"Segmentation method: {seg_method}")
print(f"Traditional method: {trad_method}")

# ----------------------
# Output directory logic
# ----------------------

def resolve_output_dir(rootfolder, user_output):
    """
    Determine the output directory path.
    Priority: user_output > rootfolder/Saved Segmentations > ~/Saved Segmentations
    """
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
    """
    Ensure directory exists and is writable.
    Falls back to home directory if primary location fails.
    """
    try:
        primary.mkdir(parents=True, exist_ok=True)
        # Test write permissions
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
print(f'Using output folder: {output_dir}')
print("=" * 80)

# ----------------------
# Processing parameters
# ----------------------
np.random.seed(42)  # The Answer to the Ultimate Question of Life, The Universe, and Everything
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
        
        # Load model architectures
        for title in custom_models_list:
            if ans_surface in title:
                with open(Path(custom_models) / title, 'r') as json_file:
                    loaded_model = model_from_json(json_file.read())
                print('Root surface model loaded')
            elif ans_tip in title:
                with open(Path(custom_models) / title, 'r') as json_file:
                    loaded_model_tip = model_from_json(json_file.read())
                print('Root tip model loaded')
        
        # Load weights
        for title in custom_weights_list:
            if ans_surface in title:
                loaded_model.load_weights(str(Path(custom_models) / title))
                print('Root surface weights loaded')
            elif ans_tip in title:
                loaded_model_tip.load_weights(str(Path(custom_models) / title))
                print('Root tip weights loaded')
    else:
        print('Default models and weights loading')
        
        if exp_type == 'Microscopy Through':
            from keras_unet.models import satellite_unet
            loaded_model = satellite_unet(input_shape=(size, size, 1))
            loaded_model_tip = satellite_unet(input_shape=(size, size, 1))
            loaded_model.load_weights('models/model_through_surface_256_sat_dice_update22092021_138.h5')
            loaded_model_tip.load_weights('models/model_through_tip_256_sat_dice_update22092021_193.h5')
        else:
            from keras_unet.models import custom_unet
            loaded_model = custom_unet(
                input_shape=(256, 256, 1),
                use_batch_norm=True,
                num_classes=1,
                filters=64,
                dropout=0.2,
                output_activation='sigmoid'
            )
            loaded_model_tip = custom_unet(
                input_shape=(256, 256, 1),
                use_batch_norm=True,
                num_classes=1,
                filters=64,
                dropout=0.2,
                output_activation='sigmoid'
            )
            loaded_model.load_weights('models/Microscopy_sandwich_256_dicecoef_customunet_01102021.h5')
            loaded_model_tip.load_weights('models/Microscopy_sandwich_TIP_256_dicecoef_customunet_01102021.h5')
    
    print('Models and weights are loaded')

# ----------------------
# Input files
# ----------------------
rootlist = [i for i in os.listdir(rootfolder) if i.endswith('.tif')]
print('The following files will be processed:')
print(rootlist)

if not rootlist:
    print("ERROR: No .tif files found in input folder!")
    raise SystemExit("No .tif files found. Exiting.")

# Calculate total loop length for progress bar
looplenght = looplenght_determinator(list_image=rootlist, folder=rootfolder)
print(f"Total frames to process: {looplenght}")
print("=" * 80)

# ----------------------
# Main processing loop
# ----------------------
for inc_root in rootlist:
    sg.theme('Light Brown 8')
    print(f"\nProcessing file: {inc_root}")
    
    # Read image stack
    test = tiff.imread(Path(rootfolder) / inc_root)
    test_shape = test.shape
    print(f"Stack shape: {test_shape}")
    
    tradlist = []
    X_test = np.zeros((len(test), size, size), dtype=np.float32)
    
    print("Converting stack to array and resizing/padding to 256x256")
    for n in range(len(test)):
        sg.OneLineProgressMeter(
            inc_root + ' import, padding, resizing',
            n + 1, len(test), 'key'
        )
        
        im = (test[n] / 255).astype('uint8')
        tradlist.append(im)
        
        # Handle non-square images
        if im.shape[0] != im.shape[1]:
            desired_size = size
            old_size = im.shape
            ratio = float(desired_size) / max(old_size)
            new_size = tuple([int(x * ratio) for x in old_size])
            img_t = cv2.resize(im, (new_size[1], new_size[0]))
            
            # Add padding
            delta_w = desired_size - new_size[1]
            delta_h = desired_size - new_size[0]
            top, bottom = delta_h // 2, delta_h - (delta_h // 2)
            left, right = delta_w // 2, delta_w - (delta_w // 2)
            color = np.average(im)
            img_t = cv2.copyMakeBorder(
                img_t, top, bottom, left, right,
                cv2.BORDER_CONSTANT, value=color
            )
            X_test[n] = img_t / 255
        else:
            X_test[n] = resize(im, (size, size), mode='constant', preserve_range=True) / 255
    
    # Save resized original stack
    path = output_dir / f"{inc_root}_original_resized256.tif"
    tiff.imwrite(str(path), X_test)
    print(f"Saved resized original: {path.name}")
    
    tradlist = np.array(tradlist)
    
    # ----------------------
    # Segmentation predictions
    # ----------------------
    if seg_method == 'Deep Machine Learning':
        X_test_exp = np.expand_dims(X_test, 3)
        
        print("Predicting root surface by deep machine learning...")
        preds_test = loaded_model.predict(X_test_exp, verbose=2, batch_size=1)
        prediction = (preds_test > 0.5).astype(np.uint8)
        
        print("Predicting root tip by deep machine learning...")
        preds_test = loaded_model_tip.predict(X_test_exp, verbose=2, batch_size=1)
        prediction_tip = (preds_test > 0.5).astype(np.uint8)
        
    elif seg_method == 'Own masks':
        print("Loading custom masks...")
        prediction = tiff.imread(Path(binary_folder) / 'surface' / inc_root)
        prediction_tip = tiff.imread(Path(binary_folder) / 'tip' / inc_root)
        
        prediction = resize(
            prediction, (len(prediction), size, size),
            mode='constant', preserve_range=True
        )
        prediction_tip = resize(
            prediction_tip, (len(prediction_tip), size, size),
            mode='constant', preserve_range=True
        )
        
        # Normalize if needed
        if np.max(prediction) > 1:
            prediction = prediction / np.max(prediction)
        if np.max(prediction_tip) > 1:
            prediction_tip = prediction_tip / np.max(prediction_tip)
            
    else:  # Traditional segmentation
        print('Traditional segmentation of root surface...')
        prediction = []
        
        if exp_type == 'Microscopy Through':
            for img_pred in X_test:
                prediction.append(trad_micro_through(img_pred.copy()))
        else:
            for img_pred in tradlist:
                prediction.append(trad_micro_sandwich(img_pred.copy()))
        
        prediction = np.array(prediction)
        
        # Use DML for tip prediction
        X_test_exp = np.expand_dims(X_test, 3)
        print("Predicting root tip by deep machine learning...")
        preds_test = loaded_model_tip.predict(X_test_exp, verbose=2, batch_size=1)
        prediction_tip = (preds_test > 0.5).astype(np.uint8)
    
    # Save predictions
    path_tip = output_dir / f"{inc_root}_root_tip_256.tif"
    tiff.imwrite(str(path_tip), prediction_tip)
    print(f"Saved root tip prediction: {path_tip.name}")
    
    path_surface = output_dir / f"{inc_root}_root_surface_256.tif"
    tiff.imwrite(str(path_surface), prediction)
    print(f"Saved root surface prediction: {path_surface.name}")
    
    # Cleanup
    gc.collect()
    sleep(0.02)

print("\n" + "=" * 80)
print("✓ Processing complete!")
print(f"⏱  Total time: {time() - start_time:.2f} seconds")
print(f"📁 Output directory: {output_dir}")
print("=" * 80)

# Easter egg
print("Um9vdCByb2NrISB6bG9vcDI3")

gc.collect()