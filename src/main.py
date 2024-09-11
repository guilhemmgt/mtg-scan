import os
import argparse
import glob
import cv2 as cv

from scanner import Scanner

from save import Save


def parse_command_line ():
    parser = argparse.ArgumentParser ()
    subparser = parser.add_subparsers (dest='command', required=True)

    subparser_scan = subparser.add_parser ('scan')   # Scan
    subparser_scan.add_argument ('-i', '--input_path', default='./in_images/', help='path containing the images to be analyzed')
    subparser_scan.add_argument ('-o', '--output_path', default='./out_images', help='output path for the results')
    subparser_scan.add_argument ('-p', '--phash_path', default='./phash.dat', help='pre-calculated phash reference file')
    subparser_scan.add_argument ('-v', '--verbose', default=False, action='store_true', help='run with verbose mode')
    subparser_scan.add_argument ('-d', '--draw', default=False, action='store_true', help='run with draw mode')

    subparser_save = subparser.add_parser ('save')   # Save
    subparser_save.add_argument ('-i', '--input_path', default='../ref_images', help='path containing the images to be referenced')
    subparser_save.add_argument ('-o', '--output_path', default='../phash.dat', help='path for output reference file')
    subparser_save.add_argument ('-v', '--verbose', default=False, action='store_true', help='run with verbose mode')
    subparser_save.add_argument ('-f', '--force_update', default=False, action='store_true', help='update cards data even if already up to date')

    args = parser.parse_args ()

    return args


def run_scan (args:argparse.ArgumentParser):
    # Reads image paths
    image_paths = glob.glob (args.input_path + "*.jpg")
    images = [cv.imread (path) for path in image_paths]
    
    # Scans obtained images
    s = Scanner (args.verbose)
    ids = [s.scan (image) for image in images]
    
    print (ids)
    
    
def run_save (args:argparse.ArgumentParser):
    s = Save (args.verbose)
    
    s.update_cards (args.force_update)
    s.update_ref_phash (True)


def main ():
    # Ensure that the current working directory is the project's root
    file_path = os.path.dirname(os.path.realpath(__file__))
    cwd = os.getcwd ()
    if (file_path == cwd):
        os.chdir ('..')
    
    # Parse command line arguments
    args = parse_command_line()

    # Action
    if (args.command == 'scan'):
        run_scan (args)
    if (args.command == 'save'):
        run_save(args)


if __name__ == "__main__":
    main ()
    