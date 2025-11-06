#preprocessing.py
#this file contains functions for preprocessing images, such as color space conversion
import cv2
import numpy as np

def rgb_to_hsv(img_array):
    # 1. De-normalize and Cast to uint8 (Necessary for cv2)
    # This undoes the ImageDataGenerator's rescale=1./255.0
    # The array becomes uint8 in range [0, 255]
    img_array_uint8 = (img_array * 255).astype(np.uint8)

    # 2. Convert Color Space (HSV values are now defined by cv2: H:[0-179], S,V:[0-255])
    hsv_array = cv2.cvtColor(img_array_uint8, cv2.COLOR_RGB2HSV)
    
    # 3. Cast back to float and Re-normalize (Necessary for the Model)
    # The array is now float in range [0.0, 1.0] (or similar, depending on HSV conversion)
    return hsv_array.astype(np.float32) / 255.0

def rgb_clahe(img_rgb_float):
    """Applies CLAHE to the V-channel (luminance) and converts back to RGB."""
    
    # 1. Convert Keras float [0, 1] to OpenCV uint8 [0, 255]
    img_uint8 = (img_rgb_float * 255).astype(np.uint8) 
    
    # 2. Convert to LAB or HSV (LAB is often preferred for preservation)
    img_lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB) 
    L, A, B = cv2.split(img_lab)

    # 3. Apply CLAHE ONLY to the L (Luminance) channel
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8)) # Use your optimized params
    L_enhanced = clahe.apply(L)

    # 4. Merge back and convert to RGB
    img_merged_lab = cv2.merge([L_enhanced, A, B])
    img_output_rgb = cv2.cvtColor(img_merged_lab, cv2.COLOR_LAB2RGB)

    # 5. Convert back to Keras float [0, 1]
    return img_output_rgb.astype(np.float32) / 255.0

