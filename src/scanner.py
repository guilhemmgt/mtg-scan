import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import imagehash
from PIL import Image as PILImage
import json
import time
import statistics
import vptree

from testimage import TestImage
from referenceimage import ReferenceImage
from preprocesser import PreProcesser
from readerwriter import ReaderWriter
from segmenter import Segmenter, Thresholding
from utils import _convex_hull_polygon, _get_bounding_quad, four_point_transform, binary_array_to_dec

class Scanner:
    verbose : bool
    rw:ReaderWriter
    
    def __init__ (self, verbose:bool=False):
        self.verbose = verbose
        self.rw = ReaderWriter (self.verbose)
    
    
    def scan (self, image:np.ndarray):
        print ("Recognizing card...")     
        
        start_time = time.time() 
        
        # Creates test image object
        test_image = TestImage (image)
        
        # Pre-process raw image
        p = PreProcesser (self.verbose)
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) # CLAHE
        p.pre_process_image (test_image, clahe)
        
        # Gets a card contour from pre-processed image
        s = Segmenter (Thresholding.ADAPTATIVE, self.verbose)
        s.segment (test_image)
        
        # Computes pHash
        phash = self.computes_phash (test_image.card_image)

        # Compares pHash
        id = self.compare_phash (phash)
        
        if (self.verbose):
            exec_time = time.time() - start_time
            print(f"\tDone in {round (exec_time, 5)} s")

        return id
    
    def computes_phash (self, image):
        if (self.verbose):
            start_time = time.time()
            print("\tComputing phash...")
        phash = imagehash.phash (PILImage.fromarray (np.uint8 (255 * cv.cvtColor (image, cv.COLOR_BGR2RGB))), hash_size=32)
        binary_phash = phash.hash
        dec_phash = binary_array_to_dec (binary_phash)
        if (self.verbose):
            exec_time = time.time() - start_time
            print(f"\t\tDone in {round (exec_time, 5)} s")
        return binary_phash.flatten ()
        
    def compare_phash (self, phash):
        ref_images = self.rw.get_references ()
        ref_images_objects = []
        
        # Converts json to ReferenceImages
        if (self.verbose):
            print ("\tConverting json to objects...")
            start_time = time.time ()
        for r in ref_images:
            ref_images_objects.append (ReferenceImage (r['id'], imagehash.hex_to_hash(r['phash']).hash.flatten ()))
        ref_images = ref_images_objects
        if (self.verbose):
            exec_time = time.time () - start_time
            print(f"\t\tDone in {round (exec_time, 5)} s")
        
        # Builds vp tree
        if (self.verbose):
            print ("\tBuilding vp tree...")
            start_time = time.time ()
        def hamming (a, b):
            return np.count_nonzero(a.phash != b.phash)
        tree = vptree.VPTree (ref_images, hamming)
        if (self.verbose):
            exec_time = time.time () - start_time
            print(f"\t\tDone in {round (exec_time, 5)} s")

        # Comparing hashes
        if (self.verbose):
            print ("\tComparing phashes...")
            start_time = time.time ()

        match = tree.get_nearest_neighbor (ReferenceImage ('', phash))

        # min_i = int (np.argmin (diff))
        # min_diff = diff[min_i]
        # min_id = ref_images[min_i].id
        
        # query = ReferenceImage ('', phash)
        # res = tree.knn (query, 1)
        if (self.verbose):
            exec_time = time.time () - start_time
            print(f"\t\tDone in {round (exec_time, 5)} s")
            
        
        # print (f"mean:{statistics.mean(diff)}, med:{statistics.median(diff)}, std:{statistics.stdev(diff)}")
        
        print("")
        print (f"{match[0]} , {match[1].id}")
        # print (exec_time / len(diff))
        # print (min_diff)
        # print (min_id)
        # std = statistics.stdev(diff)
        # mea = statistics.mean(diff)
        # for i, d in enumerate (diff):
        #     if d < mea - 4*std:
        #         print (f"{ref_images[i]['id']} with diff={d}")
        # for r in res:
        #     print (r[1])
        #     print (r[0].id)
        
        
        return match[1].id
        