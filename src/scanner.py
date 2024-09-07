import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

from testimage import TestImage
from preprocesser import PreProcesser
from segmenter import Segmenter, Thresholding
from utils import _convex_hull_polygon, _get_bounding_quad, four_point_transform

class Scanner:
    verbose : bool
    
    def __init__ (self, verbose:bool=False):
        self.verbose = verbose
    
    def scan (self, image:np.ndarray):
        # Creates test image object
        test_image = TestImage (image)
        
        # Pre-process raw image
        p = PreProcesser (self.verbose)
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) # CLAHE
        p.pre_process_image (test_image, clahe)
        
        # Gets a card contours from pre-processed image
        s = Segmenter (Thresholding.ADAPTATIVE, self.verbose)
        s.segment (test_image)
        
        #
        
        plt.figure ()
        plt.imshow (test_image.card_image)
        plt.show (block=True)