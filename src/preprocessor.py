import time
import numpy

import cv2

from inputimage import InputImage

class PreProcessor:
    verbose : bool
    
    def __init__(self, verbose:bool=False):
        self.verbose = verbose


    def pre_process_image(self, image:InputImage, clahe, max_size:int=311): #936
        '''
        Pre process test and reference images for matching
        '''
        if (self.verbose):
            print("\tPre processing...")

        start_time = time.time() # Performance stats

        if (type(image) == InputImage):
            preprocessed_image = image.raw_image # Result image
        else:
            preprocessed_image = image

        # Resize to a max of 'max_size' pixels on the longest side
        image_shape = preprocessed_image.shape
        longest_side = max(image_shape[0], image_shape[1])
        if longest_side > max_size:
            scale_factor = max_size / longest_side
            shape_scaled = (int(image_shape[1]*scale_factor), int(image_shape[0]*scale_factor))
            preprocessed_image = cv2.resize(preprocessed_image, shape_scaled)
            if (self.verbose):
                print("\t\tResizing to " + str(shape_scaled[0]) + "x" + str(shape_scaled[1]))

        # Histogram equalization (CLAHE)
        lab = cv2.cvtColor(preprocessed_image, cv2.COLOR_BGR2LAB)    # Conversion to LAB color space (lightness, redness, yellowness)
        lab[...,0] = clahe.apply(lab[...,0])                         # Apply CLAHE to lightness plane
        preprocessed_image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)    # Conversion back to BGR color space used by cv2 (blue, green, red)

        if (type(image) == InputImage):
            image.preprocessed_image = preprocessed_image

        if (self.verbose):
            exec_time = time.time() - start_time
            print(f"\t\tDone in {round (exec_time, 5)} s")
