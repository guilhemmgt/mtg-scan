import json
import traceback
import time
import os
import sys
import urllib
import requests
import pickle

class ReaderWriter:
    scryfall_bulks_url:str = 'https://api.scryfall.com/bulk-data'
    phash_filename:str = './data/phash.json'
    local_data_filename:str = './data/cards_data.json'
    local_bulk_filename:str = './data/cards_bulk.json'
    tree_filename:str = './data/tree'
    
    verbose:bool
    
    def __init__ (self, verbose=False):
        self.verbose = verbose
    
    
    def get_online_bulk (self):
        all_bulks = self._try_read_online_json (self.scryfall_bulks_url, "bulk")
        if 'data' in all_bulks:
            # return all_bulks['data'][2]
            return all_bulks['data'][3]
        else:
            return {}
    def get_online_data (self, bulk):
        return self._try_read_online_json (bulk['download_uri'], "data")
    def get_tree (self):
        if (self.verbose):
            print (f"\tRetrieving tree...")
            start_time = time.time ()
        try:
            with open (self.tree_filename, "rb") as file:
                data = file.read (data)
            if (self.verbose):
                exec_time = time.time () - start_time
                print (f"\t\tRetrieved {sys.getsizeof (data)} bytes in {round (exec_time, 5)} s.")
        except FileNotFoundError:
            data = {}
            if (self.verbose):
                print (f"\t\tFailed to retrieve tree.")
        return data
    def get_local_bulk (self):
        return self._try_read_local_json (self.local_bulk_filename, "bulk")
    def get_local_data (self):
        return self._try_read_local_json (self.local_data_filename, "data")
    def get_references (self):
        return self._try_read_local_json (self.phash_filename, "references")
    

    def write_tree (self, data):
        if (self.verbose):
            print (f"\tWriting tree...")
            start_time = time.time ()
        with open (self.tree_filename, "wb") as file:
            file.write (data)
        if (self.verbose):
            exec_time = time.time () - start_time
            print (f"\t\tWrote {sys.getsizeof (data)} bytes in {round (exec_time, 5)} s.")
    def write_bulk (self, bulk):
        return self._try_write_json (bulk, self.local_bulk_filename, "bulk")
    def write_data (self, data):
        return self._try_write_json (data, self.local_data_filename, "data")     
    def write_references (self, references):
        return self._try_write_json (references, self.phash_filename, "references")
    
    
    def _try_read_online_json (self, uri, name):
        if (self.verbose):
            print (f"\tRetrieving online {name}...")
            start_time = time.time ()
        time.sleep (0.1) # 100ms delay for good citizenship
        with urllib.request.urlopen (uri) as url:
            json_data = json.load (url)
        if (self.verbose):
            exec_time = time.time () - start_time
            count = sum (1 for c in json_data)
            print (f"\t\tRetrieved {count} items ({sys.getsizeof (json_data)} bytes) in {round (exec_time, 5)} s.")
        return json_data
    
    def _try_read_local_json (self, filename, name):
        if (self.verbose):
            print (f"\tRetrieving local {name}...")
            start_time = time.time ()
        try:
            with open (filename, 'r', encoding='utf-8') as file:
                json_data = json.load (file)
            if (self.verbose):
                exec_time = time.time () - start_time
                count = sum (1 for c in json_data)
                print (f"\t\tRetrieved {count} items ({sys.getsizeof (json_data)} bytes) in {round (exec_time, 5)} s.")
        except FileNotFoundError:
            json_data = {}
            if (self.verbose):
                print (f"\tFailed to retrieve local {name} at {filename}.")
        return json_data
    
    def _try_write_json (self, json_data, filename, name):
        if (self.verbose):
            print (f"\tWriting {name}...")
        os.makedirs (os.path.dirname (filename), exist_ok=True)
        with open (filename, 'w', encoding='utf-8') as file:
            json.dump (json_data, file, ensure_ascii=False)
        if (self.verbose):
            print (f"\tWrote {name}.")
        return True