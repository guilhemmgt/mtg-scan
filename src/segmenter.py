import time

import numpy as np
from shapely.affinity import scale
import cv2 as cv
import matplotlib.pyplot as plt

from testimage import TestImage
from cardcandidate import CardCandidate

class Thresholding:
    SIMPLE="simple"
    ADAPTATIVE="adaptative"
    RGB="simple RGB"


class Segmenter:
    verbose : bool
    threshold : Thresholding
    
    
    def __init__(self, threshold:Thresholding, verbose:bool=False):
        self.verbose = verbose
        self.threshold = threshold


    def segment (self, test_image:TestImage) -> TestImage:
        if (self.verbose):
            print("\tSegmenting...")
        
        # Extracts contours from the preprocessed image
        contours = self._contour_image (test_image)
        contours = sorted (contours, key=cv.contourArea, reverse=True)
        
        # Gets the card contours out of all contours
        # TODO better contour selection
        card_contour = contours[1]
        
        # Corrects rotation
        card_rect = cv.minAreaRect (card_contour) # ((center x, center y), (width, height), angle)
        M = cv.getRotationMatrix2D (card_rect[0], card_rect[2], 1)
        card_vertices = np.int32 (np.round (cv.transform (np.array ([cv.boxPoints (card_rect)]), M)[0]))
        print (card_vertices)
        cv.drawContours(test_image.preprocessed_image, card_contour, -1, (0,255,0), 3)
        plt.imshow (test_image.preprocessed_image)
        plt.show(block=True)
        card_image = cv.warpAffine (test_image.preprocessed_image, M, (test_image.preprocessed_image.shape[1], test_image.preprocessed_image.shape[0]))
        
        # cv2.drawContours(im, contours, -1, (0,255,0), 3)
        
        
        # Crops image
        card_image = card_image[card_vertices[1][1]:card_vertices[0][1], card_vertices[1][0]:card_vertices[2][0]]
        
        # Modifies test image
        test_image.card_image = card_image
        
        return test_image


    def _contour_image (self, test_image:TestImage) -> tuple[np.ndarray]:
        """ 
        Wrapper for contouring methods
        """
        match (self.threshold):
            case Thresholding.SIMPLE:
                return self._contour_image_simple (test_image)
            case Thresholding.ADAPTATIVE:
                return self._contour_image_adaptative (test_image)
            case Thresholding.RGB:
                return self._contour_image_rgb (test_image)
            case _:
                raise ValueError ("Unknown threshold method", self.threshold)
                

    def _contour_image_simple(self, test_image:TestImage) -> tuple[np.ndarray]:
        """
        Contours given image with simple thresholding
        """
        pp_image = test_image.preprocessed_image
        
        grayed = cv.cvtColor(pp_image, cv.COLOR_BGR2GRAY)

        _, thresholded = cv.threshold(grayed, 70, 255, cv.THRESH_BINARY)
        test_image.thresholded_image = thresholded

        contours, _ = cv.findContours(np.uint8(thresholded), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        return contours


    def _contour_image_adaptative(self, test_image:TestImage) -> tuple[np.ndarray]:
        """
        Contours given image with adaptative thresholding
        """
        pp_image = test_image.preprocessed_image

        grayed = cv.cvtColor(pp_image, cv.COLOR_BGR2GRAY)

        block_size = 1 + 2 * (min(pp_image.shape[0], pp_image.shape[1] // 20))
        thresholded = cv.adaptiveThreshold(grayed, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, block_size, 10)
        test_image.thresholded_image = thresholded

        contours, _ = cv.findContours(np.uint8(thresholded), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        return contours


    def _contour_image_rgb(self, test_image:TestImage) -> tuple[np.ndarray]:
        """
        Contours given image with simple RGB thresholding
        """
        pp_image = test_image.preprocessed_image

        blue = pp_image[..., 0]
        green = pp_image[..., 1]
        red = pp_image[..., 2]

        _, threshold_blue = cv.threshold(blue, 110, 255, cv.THRESH_BINARY)
        _, threshold_green = cv.threshold(green, 110, 255, cv.THRESH_BINARY)
        _, threshold_red = cv.threshold(red, 110, 255, cv.THRESH_BINARY)

        thresholded = cv.merge((threshold_blue, threshold_green, threshold_red))
        test_image.thresholded_image = thresholded

        contours_blue, _ = cv.findContours(np.uint8(threshold_blue), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        contours_green, _ = cv.findContours(np.uint8(threshold_green), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        contours_red, _ = cv.findContours(np.uint8(threshold_red), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        contours = contours_blue + contours_green + contours_red

        return contours
