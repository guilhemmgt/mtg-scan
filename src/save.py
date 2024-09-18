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
from collections import defaultdict
import vptree

from referenceimage import ReferenceImage
from readerwriter import ReaderWriter
from utils import binary_array_to_dec

class Save:
    
    verbose:bool
    rw:ReaderWriter
    
    def __init__ (self, verbose:bool):
        self.verbose = verbose
        self.rw = ReaderWriter (verbose)
        
    def update_tree (self, force_update_data:bool=False):
        ref = self.rw.get_references ()
        phashes = [r['phash'] for r in ref]
        tree = pynear.VPTreeBinaryIndex ()
        tree.set (phashes)
        tree_data = pickle.dumps (tree)
        self.rw.write_tree (tree_data)
        
    
    def update_ref_phash (self, force_update_data:bool=False):
        print ("Updating phashes...")
        
        local_data = self.rw.get_local_data ()

        if (self.verbose):
            print ("\tComputing cards phashes...")
        ref_images = []
        
        # 'Try' block to write already processed cards to disk even in the event of a failure
        try:
            i = 0
            for card in local_data:
                # Prints progress
                if i % 5000 == 0:
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
                    phash = imagehash.phash (PILImage.fromarray (np.uint8 (255 * cv.cvtColor (image, cv.COLOR_BGR2RGB))), hash_size=4)
                    dec_phash = binary_array_to_dec (phash.hash)
                    ref_images.append (ReferenceImage (id, str (dec_phash)).toJSON ())
        except:
            traceback.print_exc()
                
        if (self.verbose):
            print (f"\tComputed {sum (1 for c in local_data)} cards phashes.")
            
        self.rw.write_references (ref_images)
            
        return
                    
               
    def update_cards (self, force:bool=False):
        print ("Updating cards...")
        start_time = time.time ()
        
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
        verbose_filter_stats = defaultdict (int)
        for card in online_data:
            # Removes non-paper game cards (i.e. Arena/MTGO exclusive cards)
            if 'paper' not in card['games']:
                verbose_filter_stats['non_paper'] += 1
                continue
            # Removes art cards
            if 'card_faces' in card and len(card['card_faces'])==2 and card['card_faces'][0]['oracle_text']=="" and card['card_faces'][1]['oracle_text']=="":
                verbose_filter_stats['art'] += 1
                continue
            # Removes tokens
            if card['set_type'] == 'token':
                verbose_filter_stats['token'] += 1
                continue
            # Removes oversized cards
            if card['oversized'] == True:
                verbose_filter_stats['oversized'] += 1
                continue
            # Removes missing images
            if card['image_status'] == 'placeholder' or card['image_status'] == 'missing':
                verbose_filter_stats['missing_img'] += 1
                continue
            # Removes The List cards (nearly identical to an already existing card)
            if card['set'] == 'plst' or card['set'] == 'ulst':
                verbose_filter_stats['the_list'] += 1
                continue
            # Adds card
            filtered_online_data.append (card)
            
        # Groups cards with the same name
        unique_name_cards = {}
        for card in filtered_online_data:
            if card['name'] in unique_name_cards:
                unique_name_cards[card['name']].append (card)
            else:
                unique_name_cards[card['name']] = [card]
        
        # Only keeps cards who are not nearly-identical to an other card with the same name (for ex., eliminate all but 1 generic same-looking Sol Ring)
        kept_cards = []
        for name in unique_name_cards:
            same_name_cards = unique_name_cards[name]
            ok_cards = []
            for card in same_name_cards:
                ok = True
                for ok_card in ok_cards:
                    # A card is not kept if it meets all of the following criterias :
                    if (ok_card['frame'] == card['frame'] and # Same frame
                        ok_card['full_art'] == card['full_art'] and # Both (not) full art
                        ok_card['border_color'] == card['border_color'] and # Same bord color (white, back, ...)
                        ok_card['textless'] == card['textless'] and # Both (not) textless
                        (('watermark' not in ok_card and 'watermark' not in card) or ('watermark' in ok_card and 'watermark' in card and ok_card['watermark'] == card['watermark'])) and # Same watermarks
                        (('frame_effects' not in ok_card and 'frame_effects' not in card) or ('frame_effects' in ok_card and 'frame_effects' in card and ok_card['frame_effects'] == card['frame_effects'])) and # Same frame effects
                        ('illustration_id' in ok_card and 'illustration_id' in card and ok_card['illustration_id'] == card['illustration_id']) and # Same illustration
                        card['set_type'] != 'promo' and # Not a promo card
                        card['variation'] != True # Not a variation of an other card
                        ):
                        verbose_filter_stats['identical'] += 1
                        ok = False
                        break
                if (ok):
                    ok_cards.append (card)
                    kept_cards.append (card)
          
        if (self.verbose):
            for stat in list (verbose_filter_stats):
                print(f"\tRemoved {verbose_filter_stats[stat]} cards ({stat}).")
            print (f"\tKeeping {len(kept_cards)} cards.")
            
        # Writes 'bulk infos' and 'data' JSONs to disk
        self.rw.write_bulk (online_bulk)
        self.rw.write_data (kept_cards)
        
        exec_time = time.time () - start_time
        print(f"\t\tDone in {round (exec_time, 5)} s")
        
        return
