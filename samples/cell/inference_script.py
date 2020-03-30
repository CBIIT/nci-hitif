from inference import stitched_inference
from inference_utils import CleanMask
from inference import generate_inference_model
from inference import preprocess
from PIL import Image
import numpy as np
# example
# python inference_script.py /data/kimjb/Mask_RCNN/image_test/to_caltech/exp3/AUTO0218_D08_T0001F004L01A01Z01C01.tif /data/kimjb/Mask_RCNN/logs/cells20180719T1559/mask_rcnn_cells.h5. --cropsize=256 --padding=40

if __name__ == '__main__':
    import argparse
    import os 
    import sys

    parser = argparse.ArgumentParser(description='Run inference on an image for cell segmentation')

    parser.add_argument('image',
                        help='/path/to/image')
    parser.add_argument('model',
                        help='/path/to/model')
    parser.add_argument('output',
                        help='/path/to/output/image.tif')
    parser.add_argument('--cropsize', required=False,
                        default='256',
                        help='Size of patches. Must be multiple of 256')
    parser.add_argument('--padding', required=False,
                        default='40',
                        help='Amount of overlapping pixels along one axis') 
    parser.add_argument('--threshold', required=False,
                        default='30',
                        help='Min number of pixels belonging to a cell.')


    args = parser.parse_args()
    
    image = preprocess(args.image)
    padding = int(args.padding)
    cropsize = int(args.cropsize) 
    threshold = int(args.threshold)

    model = generate_inference_model(args.model, cropsize)
    import time
    start = time.time()
    stitched_inference_stack, num_times_visited = stitched_inference(image, cropsize, model, padding=padding)

#    masks = CleanMask(stitched_inference_stack, threshold, num_times_visited)
#    masks.cleanup()
#    masks.save(args.output)

    masks = CleanMask(stitched_inference_stack, threshold, )
    masks.merge_cells()
    masks.save(args.output)

    end = time.time()
    
    print('Done. Saved masks to {}. Took {} seconds'.format(args.output, end-start)) 

