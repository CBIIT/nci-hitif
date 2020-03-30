from cell import train
from imgaug import augmenters as iaa

import os
import sys
import random
import numpy as np
import cv2
import skimage.io
import warnings; warnings.simplefilter('ignore')
import time


if __name__ == '__main__':
    import configargparse
    import os
    import sys

    # Parse command line arguments
    parser = configargparse.ArgParser(default_config_files=['my_config.cfg'], description='Train Mask R-CNN to detect cells.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train model, edit this file to modify augmentations used in training'")
    parser.add_argument('--dataset', required=True,
                        metavar="/path/to/h5/datasets/",
                        help='The h5 file for nuclei datasets')
    parser.add_argument('--init', required=True,
                        metavar="Weights to initialize training",
                        help="coco, imagenet, last, random, or /path/to/weights")
    parser.add_argument('--logs', required=True,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory')
    parser.add_argument('--latest', 
                        metavar="/path/to/latest_h5",
                        help='symlink to be created for the best model')
    parser.add_argument('-c', is_config_file=True, help='config file path', 
                        metavar="/path/to/configargparse file",
                        )
    args = parser.parse_args()

    print(args)
    print("----------")
    print(parser.format_values())

    if args.init not in ['coco', 'last', 'imagenet', 'random']:
        if not os.path.exists(args.init):
            sys.exit('{} is not a valid initialization weights path'.format(args.init))
    if args.command == 'train':
        train(args.dataset, init_with=args.init, model_dir=args.logs, latest = args.latest)          
    else:
        sys.exit('{} is not a valid command'.format(args.command)) 
