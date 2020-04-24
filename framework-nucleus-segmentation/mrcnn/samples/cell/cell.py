import tensorflow as tf
tf_version = int((tf.__version__).split('.')[0])
if tf_version >= 2:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()

import os
import sys
import random
import numpy as np
import cv2
import skimage.io
import warnings; warnings.simplefilter('ignore')
import time
import h5py

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")
print(ROOT_DIR)

# Import Mask RCNN
#sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn.model import log
from skimage import measure


####################################################################
# CONFIGURATION
####################################################################

class CellsConfig(Config):
    NAME = "cells"
    
    GPU_COUNT = 1
    
    # To George and Reddy (TGAR), img/gpu could be increased to maximize training (i think I'm undersaturating the GPU so maybe we can increase this later)
    
    #GZ switching to 32 instead of 2 as the crops are 256 * 256 instead of 1024*1024
    IMAGES_PER_GPU = 8 
    
    NUM_CLASSES = 1+1 # background + cell
    
    # TGAR, change the following values based on the input size for training
    # GZ: Images are scaled to max dimension during training
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256 

    # TGAR, RPN_ANCHOR_SCALES can be decreased for smaller images. For example, the caltech images have very small cells so the following value can be decreased
    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    #TRAIN_ROIS_PER_IMAGE = 512
    
    TRAIN_ROIS_PER_IMAGE = 200

    # batch_size = num_training_data/STEPS_PER_EPOCH
    #GZ: Restore to 100, 50
    STEPS_PER_EPOCH = 10
    VALIDATION_STEPS = 2
    
    LEARNING_RATE = 1e-4 
    BATCH_SIZE =  IMAGES_PER_GPU * GPU_COUNT



####################################################################
# DATASET 
####################################################################



class CellsDataset(utils.Dataset):
    """Generates a cells dataset for training. Dataset consists of microscope images.
"""

    def generate_masks(mask_array):
        """
        Generate a dictionary of masks. The keys are instance numbers from the numpy stack and the values are the corresponding binary masks.

        Args:
            mask_array: numpy array of size [H,W]. 0 represents the background. Any non zero integer represents a individual instance

        Returns:
            Mask dictionary {instance_id: [H,W] numpy binary mask array}
        """
        masks = {} # keys are instances, values are corresponding binary mask array
        for (x,y), value in np.ndenumerate(mask_array): #go through entire array 
            if value != 0: # if cell
                if value not in masks: # if new instance introduced
                    masks[value] = np.zeros(mask_array.shape) #make new array
                dummy_array = masks[value]
                dummy_array[(x,y)] = 1
                masks[value] = dummy_array # change value of array to 1 to represent cell
        return masks
           
    def load_cells(self, h5_file, image_ids):
        """
        Loads cell images from the dataset h5 file. 
        Parameters:
        -----------
        h5_file: str
            Path to the h5 file that contains the datasets
        image_ids: numpy_array 
            The ids of the images that would be loaded
        """

        # Add class
        self.add_class("cells", 1, "cellobj")

        # Name of images / masks datasets in the h5 file.
        self.h5_file = h5_file
        self.images_dataset_name = 'DAPI_uint16touint8_normalizeandscale'
        self.masks_dataset_name = "bitmask_labeled_uint16"

        #The attribute for h5 index
        self.h5_index = 'h5_index'

        count = 0

        for _id in image_ids:
            params = {}
            params[self.h5_index] = _id
            self.add_image('cells', count, path=None, **params)
            count += 1
    
    def load_image(self, image_id):
        """
        Load the specified image from h5 file and return a [H,W,3] Numpy array.
        Parameters 
        ----------
        image_id:  int
            The id of the image in the dataset

        Returns
        -------
        numpy.ndarray[uint8][3]
        """

       
        #t1s = time.time()

        #HDF5 file with ~320K patches of 256x256. HDF5 saves data as "datasets". Note that the following datasets in the below mentioned .h5 file

        info = self.image_info[image_id]
        h5_index  = info[self.h5_index]

        with h5py.File(self.h5_file, 'r') as file_p:
            image = np.copy(file_p[self.images_dataset_name][h5_index])

        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        # If has an alpha channel, remove it for consistency
        if image.shape[-1] == 4:
            image = image[..., :3]
   
        
        #t1e = time.time()
        #print("Load image time:{0}".format(t1e-t1s))
        #print("loaded_image:{0}".format(image_id))
        return image
        

    def map_uint16_to_uint8(self, img, lower_bound=None, upper_bound=None):
        '''
        Map a 16-bit image trough a lookup table to convert it to 8-bit.

        Parameters
        ----------
        img: numpy.ndarray[np.uint16]
            image that should be mapped
        lower_bound: int, optional
            lower bound of the range that should be mapped to ``[0, 255]``,
            value must be in the range ``[0, 65535]`` and smaller than `upper_bound`
            (defaults to ``numpy.min(img)``)
        upper_bound: int, optional
           upper bound of the range that should be mapped to ``[0, 255]``,
           value must be in the range ``[0, 65535]`` and larger than `lower_bound`
           (defaults to ``numpy.max(img)``)

        Returns
        -------
        numpy.ndarray[uint8]
        '''
        
        if lower_bound is None:
            lower_bound = np.min(img)
        if not(0 <= lower_bound < 2**16):
            raise ValueError(
                '"lower_bound" must be in the range [0, 65535]')
        if upper_bound is None:
            upper_bound = np.max(img)
        if not(0 <= upper_bound < 2**16):
            raise ValueError(
                '"upper_bound" must be in the range [0, 65535]')    
        if lower_bound >= upper_bound:
            raise ValueError(
                '"lower_bound" must be smaller than "upper_bound"')
        lut = np.concatenate([
            np.zeros(lower_bound, dtype=np.uint16),
            np.linspace(0, 255, upper_bound - lower_bound).astype(np.uint16),
            np.ones(2**16 - upper_bound, dtype=np.uint16) * 255
        ])
        return lut[img].astype(np.uint8)


    
    
    def load_mask(self, image_id):
        """
        Generates instance masks for images of the given image ID.

        Parameters
        ----------
        image_id: int
            The id of the image in the class

        Return
        ------
        numpy.ndarray[n_objects, H, W] , numpy_ndarray[n_objects]
        """


        #ts = time.time()
        info = self.image_info[image_id]
        h5_index = info[self.h5_index] 

        with h5py.File(self.h5_file, 'r') as file_p:
            mask = np.copy(file_p[self.masks_dataset_name][h5_index])


        #The mask already has a different id for every nucleus
        labels = np.unique(mask)
        #Remove the background
        labels = labels[labels != 0]
        all_masks = []
        if not labels.size == 0:
            for label in np.nditer(labels):
                nucleus_mask = np.zeros(mask.shape, dtype=np.int8)
                nucleus_mask[mask == label] = 1
                all_masks.append(nucleus_mask)
        else: 
            #If there are no masks
            print("WARNING: h5_index:{0} has no masks".format(h5_index))
            nucleus_mask = np.zeros(mask.shape, dtype=np.int8)
            all_masks.append(nucleus_mask)
        mask_np = np.stack(all_masks, axis = -1).astype(np.int8)

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID, we return an array of ones

        #tf = time.time()
        #print("load_mask time:{0}".format(tf-ts))
        #print("loaded_mask:{0}".format(image_id))
        return mask_np, np.ones([len(all_masks)], dtype=np.int8) 

def get_n_images(h5_file):
    """
    Returns the number of images in the h5 file
    """
    with h5py.File(h5_file, 'r') as file_p:
        a_dataset = list(file_p.keys())[0]
        shape = file_p[a_dataset].shape
        return shape[0]


####################################################################
# TRAINING 
####################################################################

def train(h5_file, model_dir, init_with='coco',latest="latest.h5"):
    """
    Train the MRCNN using the 
    Parameters:
    -----------
    h5_file: str
        Path to the h5file that contains the ground truth datasets
    init_with: str
        Name of the h5 file to initilaze the M-RCNN network
    model_dir: str
        Directory to save logs and trained model

    lastes: src 
        The file to use as symlink for the best model
    """

    

    # Total number of images in the .h5 file

    n_images = get_n_images(h5_file) 
    print("number of images:{0}".format(n_images))
    #n_images = 200
    imgs_ind = np.arange(n_images)
    np.random.shuffle(imgs_ind)

    # Split 80-20
    train_last_id = int(n_images*0.80)

    train_indexes = imgs_ind[0:train_last_id]
    test_indexes = imgs_ind[train_last_id+1: n_images]
    n_test = len(test_indexes)
    print("Total:{0}, Train:{1}, Test:{2}".format(n_images, 
        len(train_indexes), 
        len(test_indexes))) 

    dataset_train = CellsDataset()
    dataset_train.load_cells(h5_file, train_indexes)
    dataset_train.prepare()


    dataset_test = CellsDataset()
    dataset_test.load_cells(h5_file, test_indexes)
    dataset_test.prepare()


    MODEL_DIR = model_dir
  

    config = CellsConfig()
    
    #GZ: Change to accomodate the real number of passes while
    #executing the schedule below or 200 epochs
    total_passes = 30
    n_epochs = 200
    config.STEPS_PER_EPOCH= int(train_last_id * total_passes / \
        n_epochs / config.BATCH_SIZE)

    config.VALIDATION_STEPS = int(n_test * total_passes / \
        n_epochs / config.BATCH_SIZE)


    #config.STEPS_PER_EPOCH = train_indexes.shape[0]  / config.BATCH_SIZE
    #config.VALIDATION_STEPS = test_indexes.shape[0] / config.BATCH_SIZE
    config.display()

    print("MRCNN Train module:", modellib.__file__)
    model = modellib.MaskRCNN(mode="training", config=config,
                              model_dir=model_dir)


    #print(image1.shape)
    #print( mask1.shape, ids)
    #np.save("image.npy", image1)
    #np.save("mask.npy", mask1)
    #exit()
    # Which weights to start with?
    # imagenet, coco, or last
    print('initializing with {}'.format(init_with))
    initial_layers = "heads"
    if init_with == "imagenet":
        model.load_weights(model.get_imagenet_weights(), by_name=True)
    elif init_with == "coco":
        # Load weights trained on MS COCO, but skip layers that
        # are different due to the different number of classes
        # See README for instructions to download the COCO weights

        # Local path to trained weights file
        COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
        # Download COCO trained weights from Releases if needed
        if not os.path.exists(COCO_MODEL_PATH):
            utils.download_trained_weights(COCO_MODEL_PATH)


        model.load_weights(COCO_MODEL_PATH, by_name=True,
                           exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                    "mrcnn_bbox", "mrcnn_mask"])
    elif init_with == "last":
        # Load the last model you trained and continue training
        model.load_weights(model.find_last(), by_name=True)
    elif init_with == "random":
        print("Warning: Model is initialized with random weights")
        initial_layers = "all"
    elif os.path.exists(init_with):
        import inspect
        print(inspect.getfullargspec(model.load_weights))
        print(model.load_weights.__module__)
        model.load_weights(init_with, by_name=True, reset_init_epoch=True)
    else:
        print("ERROR: No model initialization provided")
        exit(1)
        
  
    ### TRAIN THE MODEL
    # TGAR, modify how to train model. Epochs accumulate (ex. line first call to model.train means train epochs 1-75 and second call to train means train from epochs 75-100.
    #DEVICE = '/device:GPU:0'
    #with tf.device(DEVICE): 
    
    train_heads_start = time.time() 
    model.train(dataset_train, dataset_test, 
                learning_rate=config.LEARNING_RATE,
                #augmentation=augmentation, 
                epochs=75,
                layers= initial_layers)


    model.train(dataset_train, dataset_test, 
                learning_rate=config.LEARNING_RATE / 10, 
                #augmentation=augmentation, 
                epochs=100,
                layers=initial_layers)

    model.train(dataset_train, dataset_test, 
                learning_rate=config.LEARNING_RATE / 100,
                #augmentation=augmentation, 
                epochs=125,
                layers=initial_layers)


    train_heads_end = time.time()
    train_heads_time = train_heads_end - train_heads_start
    print('\n Done training {0}. Took {1} seconds'.format(initial_layers, train_heads_time))

    # Fine tune all layers
    # Passing layers="all" trains all layers. You can also 
    # pass a regular expression to select which layers to
    # train by name pattern.

    train_all_start = time.time() 

    t1s = time.time()
    model.train(dataset_train, dataset_test, 
                learning_rate=config.LEARNING_RATE / 10, 
                #augmentation=augmentation,
                epochs=150, 
                layers="all")
    t1e = time.time()
    print(t1e-t1s)

    t2s = time.time()
    model.train(dataset_train, dataset_test, 
                learning_rate=config.LEARNING_RATE / 100,
                #augmentation=augmentation,
                epochs=175, 
                layers="all")
    t2e = time.time()
    print(t2e-t2s)

    t3s = time.time()
    model.train(dataset_train, dataset_test, 
                learning_rate=config.LEARNING_RATE / 1000,
                #augmentation=augmentation,
                epochs=200, 
                layers="all")
    t3e = time.time()
    print(t3e-t3s)    

    train_all_end = time.time() 
    train_all_time = train_all_end - train_all_start
    print("Here", model.find_last())
    best_model = os.path.abspath(model.find_last())
    os.symlink(best_model, latest)

    print('\n Best model {0} symlinked to {1}'.format(best_model, latest))
    print('\n Done training all layers. Took {} seconds'.format(train_all_time))
