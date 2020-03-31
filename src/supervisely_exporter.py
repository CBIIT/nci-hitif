from generate_supervisely import generate_masks
import numpy as np


    
import argparse 
parser = argparse.ArgumentParser(description = 'Generate masks image from supervisely annotations ')
parser.add_argument('ann', help='annotation folder')
parser.add_argument('masks', help='filename for uint output image masks')
parser.add_argument('--shape', type=str, default="bitmap", help='The shape of the nucleus in supervisley. Either "bitmap" or "polygon"')

args = parser.parse_args()

ann = args.ann
import cv2
masks = generate_masks(ann, args.shape)
print("Generated {0} objects".format(np.max(masks)))
from skimage.io import imsave
imsave(args.masks, masks)
