import os
import sys
import json
import traceback
import time
from collections import defaultdict
import requests
from PIL import Image
import cv2

from img_operations import *
import utils


def update_cards(path: str = './data/hashes', hash_function: HashKind = HashKind.PHASH, n_bits: int = 64):
    print(f"Updating cards...")

    #########################################################################################
    # Step 1.
    # retrieves cards data from Scryfall

    # TODO If the local bulk is the same as the online bulk, there is no need to update anything !
    # if len(local_bulk) > 1 and not force:
    #     if local_bulk["updated_at"] == online_bulk["updated_at"]:
    #         print("\tAlready up to date.")
    #         return
    print(f"1. Downloading...")
    start_time = time.time()
    cards = download_all_cards()
    print(f"\tDone in {utils.get_elapsed_time_since(start_time)} s")

    #########################################################################################
    # Step 2.
    # reduces the amount of cards to process: get rid of irrelevant or same-looking cards
    print(f"2. Filtering...")
    start_time = time.time()
    cards = filter_cards(cards, True)
    print(f"\tDone in {utils.get_elapsed_time_since(start_time)} s")

    #########################################################################################
    # Step 3.
    # compute each card hash
    print("3. Updating phashes...")
    start_time = time.time()
    hashes = compute_phashes(cards, hash_function, n_bits)
    print(f"\tDone in {utils.get_elapsed_time_since(start_time)} s")

    #########################################################################################
    # Step 4.
    # store everything in a file or a database
    print("4. Writing to disk ...")
    start_time = time.time()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(hashes, f)
    print(f"\tDone in {utils.get_elapsed_time_since(start_time)} s")

    return


def download_all_cards():
    """
    Downloads all cards data from the latest Scryfall bulk.
    """
    bulk_url = 'https://api.scryfall.com/bulk-data'
    print(f"\tDownloading bulk infos from {bulk_url}...")
    bulk = requests.get(bulk_url).json()
    print(f"\tDownloaded {utils.get_data_size(bulk)} MB.")
    data_url = bulk['data'][2]['download_uri']
    print(f"\tDownloading cards data from {data_url}...")
    data = requests.get(data_url).json()
    print(f"\tDownloaded {utils.get_data_size(data)} MB.")
    return data


def filter_cards(cards, filter_similar):
    """
    Returns a sublist of the inputed list of cards, only keeping 'relevant' cards.
    The filtering excludes non playable MTG TCG cards (e.g. Arena, art, oversize, ...) and unusable cards (e.g. missing image).
    If filter_similar is True, very similar looking cards will be filtered out.
    """
    print(f"\tProcessing {len(cards)} cards.")
    # Stats about dismissed cards, by categories
    filter_stats = defaultdict(int)
   # Resulting cards after filtering
    relevant_cards = []

    # Filters out uninterestings items (i.e. only keeping real MTG TCG cards)
    for card in cards:
        card_lang = card["lang"]
        # HACK Only keeps french cards
        if card_lang != "en":
            filter_stats[f"HACK_non_english"] += 1
            continue
        # HACK only keeps kaladesh cards
        if card["set"] != "kld":
            filter_stats[f"HACK_non_kaladesh"] += 1
            continue
        # Removes non-paper game cards (i.e. Arena/MTGO exclusive cards)
        if "paper" not in card["games"]:
            filter_stats[f"non_paper[{card_lang}]"] += 1
            continue
        # Removes art cards
        if (
            "card_faces" in card
            and len(card["card_faces"]) == 2
            and card["card_faces"][0]["oracle_text"] == ""
            and card["card_faces"][1]["oracle_text"] == ""
        ):
            filter_stats[f"art[{card_lang}]"] += 1
            continue
        # Removes tokens
        if card["set_type"] == "token":
            filter_stats[f"token[{card_lang}]"] += 1
            continue
        # Removes oversized cards
        if card["oversized"] == True:
            filter_stats[f"oversized[{card_lang}]"] += 1
            continue
        # Removes missing images
        if card["image_status"] == "placeholder" or card["image_status"] == "missing":
            filter_stats[f"missing_img[{card_lang}]"] += 1
            continue
        # Adds card
        relevant_cards.append(card)

    if filter_similar:
        # Groups cards with the same name
        grouped_relevant_cards = {}
        for card in relevant_cards:
            if card["name"] in grouped_relevant_cards:
                grouped_relevant_cards[card["name"]].append(card)
            else:
                grouped_relevant_cards[card["name"]] = [card]

        # Only keeps cards who are not nearly-identical to an other card with the same name (e.g. eliminate all but 1 of the 40 Mike Bierek's Sol Ring reprints)
        relevant_cards = []
        for name in grouped_relevant_cards:
            same_name_cards = grouped_relevant_cards[name]
            ok_cards = []
            for card in same_name_cards:
                ok = True
                for ok_card in ok_cards:
                    # A card is discarded if it meets all of the following criterias (or if it is from The List):
                    if ((ok_card['frame'] == card['frame'] and  # Same frame
                             ok_card['full_art'] == card['full_art'] and  # Both (not) full art
                             ok_card['border_color'] == card['border_color'] and  # Same bord color (white, black, ...)
                             ok_card['textless'] == card['textless'] and  # Both (not) textless
                             (('watermark' not in ok_card and 'watermark' not in card) or ('watermark' in ok_card and 'watermark' in card and ok_card['watermark'] == card['watermark'])) and  # Same watermarks
                             (('frame_effects' not in ok_card and 'frame_effects' not in card) or ('frame_effects' in ok_card and 'frame_effects' in card and ok_card['frame_effects'] == card['frame_effects'])) and  # Same frame effects
                                 ('illustration_id' in ok_card and 'illustration_id' in card and ok_card['illustration_id'] == card['illustration_id']) and  # Same illustration
                             card['set_type'] != 'promo' and  # Not a promo card
                             card['variation'] != True)  # Not a variation of an other card
                            or (card['set'] == 'plst' or card['set'] == 'ulst')  # OR is from The List
                            ):
                        ok = False
                        filter_stats[f"similar"] += 1
                        break
                if ok:
                    ok_cards.append(card)
                    relevant_cards.append(card)

    # Stats
    for stat in list(filter_stats):
        print(f"\tRemoved {filter_stats[stat]} cards ({stat}).")
    print(f"\tKeeping {len(relevant_cards)} cards.")

    return relevant_cards


def compute_phashes(cards, hash_function, n_bits):
    """
    TODO
    """
    hashes = []
    # 'Try' block to write already processed cards to disk even in the event of a failure
    try:
        i = 0
        for card in cards:
            # Prints progress
            if i % 2000 == 0:
                print(f"\t{i} / {len(cards)} processed")
            i += 1

            id = card["id"]  # Scryfall id

            # Retrieves the card image(s) URL(s). There can be multiple images if the card is multifaced.
            img_urls = []
            if "image_uris" in card:
                img_urls.append(card["image_uris"]["normal"])
            elif "card_faces" in card:
                for face in card["card_faces"]:
                    img_urls.append(face["image_uris"]["normal"])
            else:
                print(card)
                raise Exception

            # Retrieves the card image(s).
            card_images = []
            for url in img_urls:
                # Connection can be reset or timeout unexpectedly and for unknown reasons. To counter this, each download has 10 tries.
                for _ in range(0, 10):
                    im = []
                    try:
                        # im = Image.open(requests.get(url, stream=True).raw)
                        im = cv2.imdecode(np.frombuffer(requests.get(url, stream=True).content, np.uint8), cv2.IMREAD_COLOR)
                        break
                    except:
                        pass
                if not im.any():
                    print(f"Download failed for URL {url}.")
                else:
                    card_images.append(im)

            # Computes each image's phash.
            for image in card_images:
                hash = hash_img(image, hash_function, n_bits)
                hashes.append([hash, id])
    except:
        traceback.print_exc()

    print(f"\tComputed {len(hashes)} cards phashes.")
    return hashes
