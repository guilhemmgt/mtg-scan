from imagehash import ImageHash
import json

class ReferenceImage:
    id:str
    phash:str
    
    def __init__ (self, id:str, phash:str):
        self.id = id
        self.phash = phash
    
    def toJSON (self):
        return self.__dict__
    