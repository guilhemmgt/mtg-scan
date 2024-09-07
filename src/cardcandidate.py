from shapely.geometry.polygon import Polygon
import numpy as np

class CardCandidate:
    image:np.ndarray
    bounding_poly:Polygon
    
    def __init__(self, image:np.ndarray, bounding_poly:Polygon):
        self.image = image
        self.bounding_poly = bounding_poly
