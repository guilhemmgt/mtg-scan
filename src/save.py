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

class Save:
    phash_filename:str = './data/phash.json'
    local_data_filename:str = './data/cards_data.json'
    local_bulk_filename:str = './data/cards_bulk.json'
    scryfall_bulks_url:str = 'https://api.scryfall.com/bulk-data'
    
    verbose:bool
    
    def __init__ (self, verbose:bool):
        self.verbose = verbose
    
    def update_ref_phash (self, force_update_data:bool=False):
        '''
        Updates local cards phashes.
        '''
        
        print ("Updating phashes...")
        
        
        local_data = self._get_local_data ()

        if (self.verbose):
            print ("\tComputing cards phashes...")
        ref_images = []
        
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
                
        if (self.verbose):
            print (f"\tComputed {sum (1 for c in local_data)} cards phashes.")

        if (self.verbose):
            print ("\tWriting phashes...")
        with open (self.phash_filename, 'w', encoding='utf-8') as phash_file:
            json.dump (ref_images, phash_file)
        if (self.verbose):
            print ("\tWrote phashes.")
            
        return
                    
                    
    
    def update_cards (self, force:bool=False):
        '''
        Updates cards data from Scryfall.
        '''
        print ("Updating cards...")
        
        # Retrieves local and online bulk data item for every unique card in English
        # These do NOT contains the actual cards
        online_bulk = self._get_online_bulk ()
        local_bulk = self._get_local_bulk ()

        # If the local bulk is the same as the online bulk, there is no need to update anything !
        if (len (local_bulk) > 1 and not force):
            if (local_bulk['id'] == online_bulk['id']):
                print ("\tAlready up to date.")
                return

        # Downloads cards data (the actual cards)
        online_data = self._get_online_data (online_bulk)
            
        # Filters out uninteresting cards
        filtered_online_data = []
        for card in online_data:
            # Removes non-paper game cards (i.e. Arena/MTGO exclusive cards)
            if 'paper' not in card['games']:
                continue
            # Removes art cards
            if 'card_faces' in card and len(card['card_faces'])==2 and card['card_faces'][0]['oracle_text']=="" and card['card_faces'][1]['oracle_text']=="":
                continue
            # HACK temp removes non-KLD cards
            if card['set'] != 'kld':
                continue
            # Adds card
            filtered_online_data.append (card)
        if (self.verbose):
            print (f"\tKeeping {len(filtered_online_data)} cards.")
            
        

        # Writes 'bulk infos' and 'data' JSONs to disk
        self._write_bulk (online_bulk)
        self._write_data (filtered_online_data)
        
        print ("\tDone.")
        return
            
            

    def _get_online_bulk (self):
        if (self.verbose):
            print ("\tRetrieving online bulk...")
        time.sleep (0.1) # 100ms delay for good citizenship
        try:
            all_bulks = loads (get (self.scryfall_bulks_url).text)
        except:
            traceback.print_exc ()
            default_card_bulk = {}
        else:
            default_card_bulk = all_bulks['data'][2]
            if (self.verbose):
                print (f"\tRetrieved {sys.getsizeof (default_card_bulk)} bytes.")
        return default_card_bulk
    
    def _get_online_data (self, bulk):
        if (self.verbose):
            print ("\tRetrieving online data...")
        time.sleep (0.1) # 100ms delay for good citizenship
        with urllib.request.urlopen (bulk['download_uri']) as url:
            online_data = json.load (url)
        if (self.verbose):
            count = sum (1 for c in online_data)
            print (f"\tRetrieved {sys.getsizeof (online_data)} bytes, {count} cards.")
        return online_data
       
    def _get_local_bulk (self):
        if (self.verbose):
            print ("\tRetrieving local bulk...")
        try:
            with open (self.local_bulk_filename, 'r', encoding='utf-8') as file:
                bulk = json.load (file)
            if (self.verbose):
                print (f"\tRetrieved {sys.getsizeof (bulk)} bytes.")
        except FileNotFoundError:
            bulk = {}
            if (self.verbose):
                print ("\tFailed to retrieve.")
        return bulk
    
    def _get_local_data (self):
        if (self.verbose):
            print ("\tRetrieving local data...")
        try:
            with open (self.local_data_filename, 'r', encoding='utf-8') as file:
                data = json.load (file)
            if (self.verbose):
                count = sum (1 for c in data)
                print (f"\tRetrieved {sys.getsizeof (data)} bytes, {count} cards.")
        except FileNotFoundError:
            data = {}
            if (self.verbose):
                print ("\tFailed to retrieve.")
        return data
    
    def _write_bulk (self, bulk):
        if (self.verbose):
            print ("\tWriting bulk...")
        os.makedirs (os.path.dirname (self.local_bulk_filename), exist_ok=True)
        with open (self.local_bulk_filename, 'w', encoding='utf-8') as file:
            json.dump (bulk, file, ensure_ascii=False)
        if (self.verbose):
            print ("\tWrote bulk.")
    
    def _write_data (self, data):
        if (self.verbose):
            print ("\tWriting data...")
        os.makedirs (os.path.dirname (self.local_bulk_filename), exist_ok=True)
        with open (self.local_data_filename, 'w', encoding='utf-8') as file:
            json.dump (data, file, ensure_ascii=False)
        if (self.verbose):
            print ("\tWrote data.")