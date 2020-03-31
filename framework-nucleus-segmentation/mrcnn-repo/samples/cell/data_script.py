from data import np_to_imgmask
from data import np_to_img
from data import train_test_split 
from data import reset
import os

# example line
# python data_script.py imgmask --outdir=/data/kimjb/Mask_RCNN/image_test/images --images=/data/kimjb/Mask_RCNN/image_test/image_01.npy --masks=/data/kimjb/Mask_RCNN/image_test/masks_01.npy

# python data_script.py split --outdir=/data/kimjb/Mask_RCNN/image_test/images
# python data_script.py reset --outdir=/data/kimjb/Mask_RCNN/image_test/images

if __name__ == '__main__':
    import argparse
    TRAIN_PERCENT = .9
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate data, train/test split, reset train/test splits')
    parser.add_argument("command",
                        metavar="<command>",
                        help="either 'imgmask', 'img', 'split', 'reset'")
    parser.add_argument('--outdir', required=True,
                        metavar="/path/to/output/directory/",
                        help='Output directory of the np images')
    parser.add_argument('--images', required=False,
                        metavar="/path/to/numpy/image/file",
                        help='Path to numpy image file')
    parser.add_argument('--masks', required=False,
                        metavar="/path/to/numpy/mask/file",
                        help='Path to numpy mask file')
    args = parser.parse_args()

    
    assert args.command in ['imgmask', 'img', 'split', 'reset'], "command must be in ['imgmask', 'img', 'split', 'reset']"

    train_directory = os.path.join(args.outdir, 'training_data')
    test_directory = os.path.join(args.outdir, 'testing_data')
         
    if args.command == 'imgmask':
        assert [args.images, args.masks], 'imgmask requires the --images and --masks arguments'
        np_to_imgmask(args.images, args.masks, args.outdir)   
    
    if args.command == 'img':
        assert args.images, "img requires the --images argument"    
        np_to_img(args.images, args.outdir) 
    
    if args.command == 'split':
        train_test_split(train_directory, test_directory, args.outdir, TRAIN_PERCENT)

    if args.command == 'reset':
        reset(train_directory, test_directory, args.outdir)
       
     
