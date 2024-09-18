import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import imagehash
from PIL import Image as PILImage
import json
import time
import statistics
import vptree

from inputimage import InputImage
from referenceimage import ReferenceImage
from preprocessor import PreProcessor
from readerwriter import ReaderWriter
from segmenter import Segmenter, Thresholding
from utils import _convex_hull_polygon, _get_bounding_quad, four_point_transform, binary_array_to_dec

class Scanner:
    verbose : bool
    rw:ReaderWriter
    pp:PreProcessor
    seg:Segmenter
    clahe=None
    tree=None
    
    def __init__ (self, verbose:bool=False):
        self.rw = ReaderWriter (verbose)
        self.pp = PreProcessor (verbose)
        self.seg = Segmenter (Thresholding.ADAPTATIVE, verbose)
        self.clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.verbose = verbose
        
        self.build_tree ()
    
    
    def scan (self, img:np.ndarray):
        print ("Recognizing card...")     
        
        start_time = time.time() 
        
        # Creates test image object
        input_img_obj = InputImage (img)
        
        # Pre-process raw image
        self.pp.pre_process_image (input_img_obj, self.clahe)
        
        # Gets a card contour from pre-processed image
        self.seg.segment (input_img_obj)
        
        # Computes pHash
        phash = self.computes_phash (input_img_obj.card_image)

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
        if (self.verbose):
            exec_time = time.time() - start_time
            print(f"\t\tDone in {round (exec_time, 5)} s")
        return binary_phash.flatten ()
        
        
    def compare_phash (self, phash):
        if (self.verbose):
            print ("\tComparing phashes...")
            start_time = time.time ()
        match = self.tree.get_nearest_neighbor (ReferenceImage ('', phash))
        if (self.verbose):
            exec_time = time.time () - start_time
            print(f"\t\tDone in {round (exec_time, 5)} s")
        return match[1].id


    def build_tree (self):
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
        self.tree = vptree.VPTree (ref_images, hamming)
        if (self.verbose):
            exec_time = time.time () - start_time
            print(f"\t\tDone in {round (exec_time, 5)} s")
