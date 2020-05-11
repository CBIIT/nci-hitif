import tensorflow as tf
tf_version = int((tf.__version__).split('.')[0])
if tf_version >= 2:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()

import numpy as np
import time
from cell import CellsConfig
import mrcnn.model as modellib
import skimage.io
import sys
import skimage
from PIL import Image
#from libtiff import TIFF
import time


def generate_inference_model(model_path, cropsize):
    """
    Generates an inference model from the model_path. cropsize is how big of a patch to run inference on.
    """
    import math
    class InferenceConfig(CellsConfig):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1

        # comment below if running inference on small crops
        TRAIN_ROIS_PER_IMAGE = 2000
        POST_NMS_ROIS_INFERENCE = 10000
        DETECTION_MAX_INSTANCES = 200

        #TRAIN_ROIS_PER_IMAGE = 4000 * 4
        #POST_NMS_ROIS_INFERENCE = 20000 * 4
        #DETECTION_MAX_INSTANCES = 400 * 4
        #DETECTION_NMS_THRESHOLD = 0.35
        IMAGE_MIN_DIM = cropsize #math.ceil(mindim / 256) * 256
        IMAGE_MAX_DIM = cropsize #math.ceil(maxdim / 256) * 256

    inference_config = InferenceConfig()
    print("MRCNN take from:", modellib.__file__)

    # Recreate the model in inference mode
    DEVICE = '/device:GPU:0'
    with tf.device(DEVICE):
        model = modellib.MaskRCNN(mode="inference", 
                              config=inference_config,
                              model_dir=model_path)


    # Get path to saved weights
    # Either set a specific path or find last trained weights
    # model_path = os.path.join(ROOT_DIR, ".h5 file name here")

    #model_path = model.find_last()[1]
    #model_path = '/data/kimjb/Mask_RCNN_original/logs/cells20180628T1527/mask_rcnn_cells_0100.h5'

    # Load trained weights (fill in path to trained weights here)
    assert model_path != "", "Provide path to trained weights"
    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)
    
    return model



def run_inference(model, image):
    """
    Runs inference on an image using model and returns the mask stack.
    """
    # get results
    start = time.time()
    results = model.detect([image], verbose=1)
    end = time.time()
    print("Inference took {}".format(end-start))
    r = results[0]
    masks = r['masks']
    print(masks.shape)
    return masks


def mask_stack_to_single_image(masks, checkpoint_id):
    """
    Merge a stack of masks containing multiple instances to one large image.

    Args:
        image: full fov np array
        masks: stack of masks of shape [h,w,n]. Note that image.shape != masks.shape, because the shape of the masks is the size of the inference call. Since we are doing inference in patches, the masks are going to be of size of the patch.

    Returns:
        image that is the same shape as the original raw image, containing all of the masks from the mask stack
    """
    image = np.zeros((masks.shape[0:2]),dtype=np.uint16) # image that contains all cells (0 bg, >0 is cell id)
    
    # switch shape to [num_masks, h, w] from [h, w, num_masks]
    masks = masks.astype(np.uint16) 
    #masks = np.moveaxis(masks, 0, -1)
    masks = np.moveaxis(masks, -1, 0) 

    #image_shape = masks.shape
    #image = np.zeros(image_shape[1:])
    #print(image.shape)
    num_masks = masks.shape[0] # shape = [n, h, w]
       
    print("Sum = %", np.sum(masks))
    for i in range(num_masks):
        current_mask = masks[i]
        #print(np.max(current_mask), np.sum(current_mask))
        image[current_mask > 0] = checkpoint_id
        #image = add_mask_to_ids(image, current_mask, checkpoint_id)
        checkpoint_id += 1
  
    
    print("Sum=", np.sum(image > 0))
    return image, checkpoint_id



def add_mask_to_ids(image, mask, fill_int):
    """
    Same as mask_stack_to_single_image but is just a helper function. Merges one mask from the stack into image. Gives unique id to each mask
    """
    for (row,col), value in np.ndenumerate(mask):
        if value != 0 and image[row,col] == 0:
            image[row, col] = fill_int    
    return image
    

def pad(arrays, reference, offsets):
    """
    array: Array to be padded
    reference: Reference array with the desired shape
    offsets: list of offsets (number of elements must be equal to the dimension of the array)
    """
    # Create an array of zeros with the reference shape
    result = np.zeros((reference[0],reference[1]), dtype=np.uint16)
    print('result:')
    print(result.shape)
    # Create a list of slices from offset to offset + shape in each dimension
    insertHere = [slice(offsets[dim], offsets[dim] + arrays.shape[dim]) for dim in range(arrays.ndim)]
    #print(insertHere)
    #print(arrays.shape)
    # Insert the array in the result at the specified offsets
    result[insertHere] = arrays
    return result


def stitched_inference(image, cropsize, model, padding=40):#, minsize=100):
    """
    array: Array to be padded
    reference: Reference array with the desired shape
    offsets: list of offsets (number of elements must be equal to the dimension of the array)
    """
    stack = np.zeros((image.shape[0], image.shape[1],1),dtype=np.uint16) # make new image of zeros (exclude third dimension, not using rgb)
    visited = np.zeros(image.shape[0:2])
    num_times_visited = np.zeros(image.shape[0:2])
    
    num_row = image.shape[0] # num rows in the image
    num_col = image.shape[1]
    print(image.shape)
    
    assert cropsize < num_row and cropsize < num_col, 'cropsize must be smaller than the image dimensions'
        
    #rowlist = np.concatenate(([0],np.arange(cropsize-padding, num_row, cropsize)))
    #collist = np.concatenate(([0],np.arange(cropsize-padding, num_col, cropsize)))
    checkpoint_id = 1
    for row in np.arange(0, num_row, cropsize-padding): # row defines the rightbound side of box
        for col in np.arange(0, num_col, cropsize-padding): # col defines lowerbound of box
            masks_with_ids = np.zeros(image.shape[0:2])
            upperbound = row
            lowerbound = row + cropsize
            leftbound  = col
            rightbound = col + cropsize
            
            if lowerbound > num_row:
                lowerbound = num_row
                upperbound = num_row-cropsize
            
            if rightbound > num_col:
                rightbound = num_col
                leftbound  = num_col-cropsize
            #upperbound = bound(final_image, cropsize, padding, minsize, row, 'upper')
            #lowerbound = bound(final_image, cropsize, padding, minsize, row, 'lower')
            #rightbound = bound(final_image, cropsize, padding, minsize, col, 'right')
            #leftbound  = bound(final_image, cropsize, padding, minsize, col, 'left')
            #print(row)
            #print(col)
            print('bounds:')
            print('upper: {}'.format(upperbound))
            print('lower: {}'.format(lowerbound))
            print('left : {}'.format(leftbound))
            print('right: {}'.format(rightbound))
            
            num_times_visited[upperbound:lowerbound, leftbound:rightbound] += 1
            cropped_image = image[upperbound:lowerbound, leftbound:rightbound, :]
            #print('cropped image shape: {}'.format(cropped_image.shape))
            #print(cropped_image.shape)
            
            masks = run_inference(model, cropped_image)
            
            #padded_masks = pad(masks, [num_row, num_col, masks.shape[2]], [upperbound, leftbound])
            #padded_masks = pad(masks, [num_row, num_col, masks.shape[2]], [upperbound,leftbound,0])
            #print('mask shape:')
            #print (padded_masks.shape)
            
            one_inference_mask_image, checkpoint_id = mask_stack_to_single_image(masks, checkpoint_id) # works
            padded_inference_mask = pad(one_inference_mask_image, [num_row, num_col], [upperbound,leftbound])
            padded_inference_mask = np.expand_dims(padded_inference_mask, axis=2)
            stack = np.concatenate((stack, padded_inference_mask), axis=2)
                        
    return stack, num_times_visited


class CleanMask():
    def __init__(self, stack, threshold, num_times_visited):
        ## input 
        self.stack = stack #stack of masks
        self.num_times_visited = num_times_visited #how many times inference ran on each pixel
        
        self.num_row = self.stack.shape[0]
        self.num_col = self.stack.shape[1]
        self.masks = np.zeros((self.num_row, self.num_col), dtype=np.uint16)
        #self.visitedPoints = np.zeros((self.num_row, self.num_col))
        self.id = 1
        self.dict = {}
        self.unique = {}
        self.cells = {}
        self.threshold = threshold
        
    def getMasks(self):
        return self.masks
    
    def visited(self, row, col):
        if self.visitedPoints[row,col] == 1:
            return True
    
    def get_ids(self, array):
        ids = set()
        for value in array:
            if value > 0:
                ids.add(value)
        frozen_set_ids = frozenset(ids)
        return frozen_set_ids
    
    def at_least_one(self, values, ids):
        for value in values:
            if value in ids:
                return True
        return False
  

    def build_connectivity_matrix(self):
        """
        Build the connectivity matrix between cells ids
        """
        start = time.time()
        n_labels = np.max(self.stack) 
        self.conn_mat = np.zeros((n_labels + 1, n_labels + 1))

    
        cells = self.stack > 0
        #Find where a pixel has more than one inference
        n_inference = np.sum(cells,2)
        to_connect = (n_inference > 1)
        indexes = np.nonzero(to_connect)
        to_process = stack[indexes[0], indexes[1], :]
        for i in range(0, to_process.shape[0]):
            
            #by profiling, working with list was faster than numpy arrays
            ids = np.unique(to_process[i,:]).tolist()
            if 0 in ids:
                ids.remove(0)
            ids_size = len(ids) 
            for i in range(ids_size - 1): 
                for j in range(i+1,ids_size):
                    self.conn_mat[ids[i], ids[j] ] += 1

   
        #The connectivity matrix is symetrical 
        conn_t = self.conn_mat.transpose()
        self.conn_mat += conn_t
        end = time.time()
        print("connectivity time = {}".format(end-start))
        

    def merge_cells(self):
        """
        Find connected cells between multiple inferences and merge strongly connected 
        cells that have overlap more than a threshold.
        """
        self.build_connectivity_matrix()
        
        #Filter out week connections
        self.conn_matrix[self.conn_matrix < self.threshold] = 0

        #Get connected components 
        np.fill_diagonal(self.conn_matrix, 1)

        from scipy.sparse.csgraph import csgraph_from_dense, connected_components
        n_conn_comp, graph_labels =  connected_components(conn_matrix, False) 

        print(n_conn_comp)
        print(graph_labels)


            


    def cleanup(self):
        for row in range(self.num_row):
            for col in range(self.num_col):
                if self.num_times_visited[row,col] > 1:
                    set_of_ids = self.get_ids(self.stack[row,col,:])
                    if len(set_of_ids) > 1:
                        if set_of_ids not in self.dict.keys():
                            self.dict[set_of_ids] = 0
                        self.dict[set_of_ids] += 1            
           
         
        oneinf = {}
        for row in range(self.num_row):
            for col in range(self.num_col):
                if self.num_times_visited[row,col] == 1:
                    set_of_ids = self.get_ids(self.stack[row,col,:])
                    if len(set_of_ids) > 0:
                        if set_of_ids not in oneinf.keys():
                            oneinf[set_of_ids] = 0
                        oneinf[set_of_ids] += 1
        
        for set_of_ids in oneinf:
            if oneinf[set_of_ids] >= self.threshold:
                self.unique[set_of_ids] = self.id
                self.id += 1
       
        #Merge similar ids
        for set_of_ids in self.dict:
            if len(set_of_ids) > 1 and self.dict[set_of_ids] >= self.threshold:
                self.unique[set_of_ids] = self.id
                self.id += 1
                for cell_id in set_of_ids:
                    temp = frozenset([cell_id])
                    if temp in oneinf.keys():
                        if oneinf[temp] >= self.threshold:
                            self.unique[temp] = self.unique[set_of_ids]
        #Change to unique ids                    
        for row in range(self.num_row):
            for col in range(self.num_col):
                set_of_ids = self.get_ids(self.stack[row,col,:])
                if len(set_of_ids) > 0:
                    if set_of_ids in self.unique.keys():
                        self.masks[row,col] = self.unique[set_of_ids]
                        
                

       
    def save(self, save_path):
        np.save('inference-stack', self.stack)
        return
        tiff = TIFF.open(save_path, mode='w')
        tiff.write_image(self.getMasks())
        tiff.close()

def map_uint16_to_uint8(img, lower_bound=None, upper_bound=None):
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


def preprocess(img):
    image = Image.open(img)
    imarray = np.array(image)
    image = skimage.color.gray2rgb(imarray)
    image = map_uint16_to_uint8(image)
    return image


    
   
