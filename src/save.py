import sys
import urllib.request
from requests import get
from json import loads
import json
import time
import traceback
import imagehash
from PIL import Image as PILImage
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

from referenceimage import ReferenceImage
from readerwriter import ReaderWriter

class Save:
    
    verbose:bool
    rw:ReaderWriter
    
    def __init__ (self, verbose:bool):
        self.verbose = verbose
        self.rw = ReaderWriter (verbose)
    
    def update_ref_phash (self, force_update_data:bool=False):
        '''
        Updates local cards phashes.
        '''
        
        print ("Updating phashes...")
        
        local_data = self.rw.get_local_data ()

        if (self.verbose):
            print ("\tComputing cards phashes...")
        ref_images = []
        
        try:
            i = 0
            for card in local_data:
                if i != 0 and i % 10 == 0:
                    print (f"\t{i} / {len (local_data)}")
                i += 1
                
                id = card['id'] # Scryfall id

                # Retrieves the card's image. There can be multiple images if the card is multifaced.
                card_images = []
                if 'image_uris' in card:
                    card_images.append (plt.imread (urllib.request.urlopen (card['image_uris']['large']), format='jpg'))
                elif 'card_faces' in card:
                    for face in card['card_faces']:
                        card_images.append ( plt.imread (urllib.request.urlopen (face['image_uris']['large']), format='jpg') )
                else:
                    print (card)
                    raise Exception
                
                # TODO: add clahe to card_images ?
                
                # Computes each image's phash.
                for image in card_images:
                    phash = imagehash.phash (PILImage.fromarray (np.uint8 (255 * cv.cvtColor (image, cv.COLOR_BGR2RGB))), hash_size=32)
                    ref_images.append (ReferenceImage (id, str (phash)).toJSON ())
        except:
            traceback.print_exc()
                
        if (self.verbose):
            print (f"\tComputed {sum (1 for c in local_data)} cards phashes.")
            
        self.rw.write_references (ref_images)
            
        return
                    
                    
    
    def update_cards (self, force:bool=False):
        '''
        Updates cards data from Scryfall.
        '''
        print ("Updating cards...")
        
        # Retrieves local and online bulk data item for every unique card in English
        # These do NOT contains the actual cards
        online_bulk = self.rw.get_online_bulk ()
        local_bulk = self.rw.get_local_bulk ()

        # If the local bulk is the same as the online bulk, there is no need to update anything !
        if (len (local_bulk) > 1 and not force):
            if (local_bulk['id'] == online_bulk['id']):
                print ("\tAlready up to date.")
                return

        # Downloads cards data (the actual cards)
        online_data = self.rw.get_online_data (online_bulk)
            
        # Filters out uninteresting cards
        filtered_online_data = []
        for card in online_data:
            # Removes non-paper game cards (i.e. Arena/MTGO exclusive cards)
            if 'paper' not in card['games']:
                continue
            # Removes art cards
            if 'card_faces' in card and len(card['card_faces'])==2 and card['card_faces'][0]['oracle_text']=="" and card['card_faces'][1]['oracle_text']=="":
                continue
            # Removes 'Teach by Example' special art card, which for whatever reason have an oracle text (and no image on Scryfall) 
            if card['id'] == 'aaf8b02f-caa9-4e4f-9ade-d02a48555550':
                continue
            # HACK temp removes non-kaladesh non-doublemasters cards
            # if card['set'] != 'kld' and card['set'] != '2xm':
                # continue
            # Adds card
            filtered_online_data.append (card)
        if (self.verbose):
            print (f"\tKeeping {len(filtered_online_data)} cards.")
            
        # Writes 'bulk infos' and 'data' JSONs to disk
        self.rw.write_bulk (online_bulk)
        self.rw.write_data (filtered_online_data)
        
        print ("\tDone.")
        return
