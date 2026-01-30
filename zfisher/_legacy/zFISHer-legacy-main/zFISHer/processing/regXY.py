import os
import numpy as np
#import zFISHer.config.config_manager as cfgmgr
from PIL import Image, ImageTk
from pystackreg import StackReg
import cv2
import zFISHer.utils.config as cfg
from zFISHer.gui.logger import Logger
from skimage.registration import phase_cross_correlation
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
from scipy.ndimage import shift


"""
Algorithm for simple 2D XY registration using two MIPs from the input files.
"""


def start_regXY(f1_c,f2_c):
    """
    File 
    """
    f1_reg_img = load_reg_img(f1_c,1)
    f2_reg_img = load_reg_img(f2_c,2)
    f1_reg_img_n = normalize_img(f1_reg_img)
    f2_reg_img_n = normalize_img(f2_reg_img)
    f1_reg_img_gs = grayscale_img(f1_reg_img_n)
    f2_reg_img_gs = grayscale_img(f2_reg_img_n)
    offset = registerXY(f1_reg_img_gs,f2_reg_img_gs)
    #register = RegisterXY()

def load_reg_img(img_c,f_num):
    if f_num ==1:
        img_path = cfg.F1_C_MIP_DIR_DICT[img_c]
    else:
        img_path = cfg.F2_C_MIP_DIR_DICT[img_c]

    file = os.listdir(img_path)[0]
    full_path = os.path.join(img_path,file)
    image = Image.open(full_path)

    return image

def normalize_img(img):
    # Convert the image to a numpy array with float32 for better precision
    img_array = np.array(img).astype(np.float32)
    
    # Check for invalid values (NaN or inf)
    if not np.all(np.isfinite(img_array)):
        Logger.warning("Input image contains NaN or inf values. Replacing with 0.")
        img_array = np.nan_to_num(img_array, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Find the minimum and maximum pixel values
    min_val = img_array.min()
    max_val = img_array.max()
    
    # Normalize the image to the range [0, 1]
    if max_val == min_val:
        Logger.warning("Image has no dynamic range (max == min). Setting normalized array to zeros.")
        normalized_array = np.zeros_like(img_array, dtype=np.float32)
    else:
        normalized_array = (img_array - min_val) / (max_val - min_val)
    
    # Ensure normalized_array is in [0, 1] and finite
    normalized_array = np.clip(normalized_array, 0.0, 1.0)
    normalized_array = np.nan_to_num(normalized_array, nan=0.0, posinf=1.0, neginf=0.0)
    
    # Scale to [0, 65535] and cast to uint16
    normalized_array = (normalized_array * 65535).astype(np.uint16)
    
    # Convert back to PIL Image
    normalized_img = Image.fromarray(normalized_array)
    
    return normalized_img


def grayscale_img(image):
# Convert PIL Image to numpy array
    image_array = np.array(image)
    
    # Check if the image has 3 dimensions (indicating an RGB image)
    if image_array.ndim == 3:
        # Convert RGB to grayscale by averaging the channels
        image_array = np.mean(image_array, axis=2)
    
    return image_array

def registerXY(img_1, img_2):
    # Perform translation transformation
    sr = StackReg(StackReg.TRANSLATION)
    reg = sr.register_transform(img_1, img_2)

    # Clip the registered result to ensure no negative values
    reg = reg.clip(min=0)

    offset = sr.get_matrix()[:2, 2]
    
    offset[0] = (-offset[0])
    offset[1] = (-offset[1])


   # cfgmgr.set_config_value("X_OFFSET",offset[0])
   # cfgmgr.set_config_value("Y_OFFSET",offset[1])

    cfg.OFFSET_X = offset[0]
    cfg.OFFSET_Y = offset[1]

    return offset
