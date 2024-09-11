import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import imagehash
from PIL import Image as PILImage
import json
import time
import statistics

from testimage import TestImage
from referenceimage import ReferenceImage
from preprocesser import PreProcesser
from readerwriter import ReaderWriter
from segmenter import Segmenter, Thresholding
from utils import _convex_hull_polygon, _get_bounding_quad, four_point_transform

class Scanner:
    verbose : bool
    rw:ReaderWriter
    
    def __init__ (self, verbose:bool=False):
        self.verbose = verbose
        self.rw = ReaderWriter (self.verbose)
    
    def scan (self, image:np.ndarray):
        print ("Recognizing card...")     
        
        start_time = time.time() # Performance stats   
        
        # Creates test image object
        test_image = TestImage (image)
        
        # Pre-process raw image
        p = PreProcesser (self.verbose)
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) # CLAHE
        p.pre_process_image (test_image, clahe)
        
        # Gets a card contours from pre-processed image
        s = Segmenter (Thresholding.ADAPTATIVE, self.verbose)
        s.segment (test_image)
        
        # Computes pHash
        if (self.verbose):
            phash_time = time.time() # Performance stats
            print("\tComputing phash...")
        phash = imagehash.phash (PILImage.fromarray (np.uint8 (255 * cv.cvtColor (test_image.card_image, cv.COLOR_BGR2RGB))), hash_size=32)
        if (self.verbose):
            exec_time = time.time() - phash_time
            print(f"\t\tDone in {round (exec_time, 5)} s")
        
        id = self.compare_phash (phash)
        
        if (self.verbose):
            exec_time = time.time() - start_time
            print(f"\tDone in {round (exec_time, 5)} s")
        
        # Compares pHashes
        return id
        
    def compare_phash (self, phash):
        ref_images = self.rw.get_references ()
            
        if (self.verbose):
            print ("\tComparing phashes...")
            start_time = time.time ()
            
        diff = []
        for ref_image in ref_images:
            ref_phash = imagehash.hex_to_hash (ref_image['phash'])
            diff.append (phash - ref_phash)
            
        min_i = int (np.argmin (diff))
        min_diff = diff[min_i]
        min_id = ref_images[min_i]['id']
        
        if (self.verbose):
            exec_time = time.time () - start_time
            print(f"\t\tDone in {round (exec_time, 5)} s")
        
        print (f"mean:{statistics.mean(diff)}, med:{statistics.median(diff)}, std:{statistics.stdev(diff)}")
        
        print("")
        print (exec_time / len(diff))
        print (min_diff)
        print (min_id)
        std = statistics.stdev(diff)
        mea = statistics.mean(diff)
        for i, d in enumerate (diff):
            if d < mea - 4*std:
                print (f"{ref_images[i]['id']} with diff={d}")
        
        return min_id
        