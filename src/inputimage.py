import numpy as np

class InputImage:

    raw_image:np.ndarray
    """ Original input image """
    
    preprocessed_image:np.ndarray
    """ Whole image, resized and equalized """
    
    thresholded_image:np.ndarray
    """ Whole image, for finding contours """
    
    card_image:np.ndarray
    """ Cropped image, for pHash """
    
    def __init__(self, raw_image):
        self.raw_image = raw_image
        