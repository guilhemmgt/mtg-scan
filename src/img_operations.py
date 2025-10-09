import time
import numpy as np
import cv2
import imagehash
import PIL
from enum import Enum

def pre_process_img(image:np.ndarray, clahe:cv2.CLAHE, verbose:bool=False) -> np.ndarray:
    if (verbose):
        print("Pre processing...")
        start_time = time.time()
        
    # Histogram equalization (CLAHE)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)  # Conversion to LAB color space
    lab[...,0] = clahe.apply(lab[...,0])        # Apply CLAHE to lightness plane
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)  # Conversion back to BGR color space used by cv2

    if (verbose):
        exec_time = time.time() - start_time
        print(f"\tDone in {round (exec_time, 5)} s")
        
    return image


class HashKind(Enum):
    DHASH=imagehash.dhash
    PHASH=imagehash.phash
    
def hash_img(image:np.ndarray, hash_kind:HashKind, n_bits:int, max_size:int=1000, verbose:bool=False) -> int:
    if (verbose):
        print("Hashing...")
        start_time = time.time()
        
    # Resize to a max of 'max_size' pixels on the longest side
    image_shape = image.shape
    longest_side = max(image_shape[0], image_shape[1])
    if longest_side > max_size:
        scale_factor = max_size / longest_side
        shape_scaled = (int(image_shape[1]*scale_factor), int(image_shape[0]*scale_factor))
        image = cv2.resize(image, shape_scaled)
        if (verbose):
            print(f"\tResizing to {str(shape_scaled[0])}x{str(shape_scaled[1])}")
    
    # Compute hash
    image = PIL.Image.fromarray(image)
    image_hash = hash_kind(image)
    
    # Converts hex hash to 'n_bits' bits signed integer hash
    hex_str_hash = str(image_hash)
    int_hash = int(hex_str_hash, 16)
    if int_hash & (1 << (n_bits-1)):
        int_hash -= 1 << n_bits
    
    if (verbose):
        exec_time = time.time() - start_time
        print(f"\tDone in {round (exec_time, 5)} s")
        
    return int_hash
