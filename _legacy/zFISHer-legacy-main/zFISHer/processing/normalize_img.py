import numpy as np
from PIL import Image

import cv2





def normalize_mip_main(img_path):
    img = load_image(img_path)
    img_n = normalize_image_8b(img)
    return img_n


def load_image(img_path):
    """
    Opens image file from given target path

    Args:
    img_path: string = image path

    Returns:
    img: * = file by given extension
    """

    img = Image.open(img_path)
    return img


def normalize_image_8b(img):
    """
    Nomalizes image intensity in 8-bit

    Args:
    img: (imgtype) = input image

    Returns:
    img_n (imgtype) = normalized input image
    """
    # Convert the image to a numpy array
    image_array = np.array(img).astype(np.float32)
    
    # Find the minimum and maximum pixel values
    min_val = image_array.min()
    max_val = image_array.max()
    
    # Normalize the image to the range [0, 1]
    normalized_array = (image_array - min_val) / (max_val - min_val)
    
    # Optionally, scale to [0, 255] for 8-bit representation
    normalized_array = (normalized_array * 255).astype(np.uint8)
    
    # Convert back to PIL Image
    img_n = Image.fromarray(normalized_array)
    
    return img_n