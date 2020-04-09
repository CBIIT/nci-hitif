import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.insert(1, './samples/cell')
from inference_utils import CleanMask
from inference import *
import matplotlib.pyplot as plt
from skimage.color import label2rgb
import skimage
from PIL import Image
import configparser
import glob
import configparser
import numpy as np
import skimage
from PIL import Image

# Parser
config_file_path="./demo_config.ini"
config = configparser.ConfigParser()
config.read(config_file_path)
param = {s: dict(config.items(s)) for s in config.sections()}['general']
for key in param:
    param[key] = int(param[key])

cropsize=param['cropsize']
padding=param['padding']
threshold=param['threshold']

# Read Image files and preprocess
img_path = "./input/*.tif"
image_list = glob.glob(img_path)
img = np.zeros((len(image_list),1078,1278))
mask = np.zeros((len(image_list),1078,1278))
# for i in range(len(image_list)):
i = 0
image = Image.open(image_list[i])
imarray = np.array(image)
image = skimage.color.gray2rgb(imarray)
image = map_uint16_to_uint8(image)

# Read Trained model
model = generate_inference_model("../model/mask_rcnn_cells_0194.h5", cropsize)

stitched_inference_stack, num_times_visited = stitched_inference(image, cropsize, model, padding=padding)
masks = CleanMask(stitched_inference_stack, threshold,num_times_visited)
masks.merge_cells()

my_mask = masks.getMasks().astype("int16")
mask[i,:,:] = my_mask
# masks.save("./output.tif")
pf, axarr = plt.subplots(1,1)
axarr.imshow(label2rgb(my_mask,bg_color=(0, 0, 0),bg_label=0))

####

stitched_inference_stack, num_times_visited = stitched_inference(image, cropsize, model, padding=padding)
masks = CleanMask(stitched_inference_stack, threshold, )
masks.merge_cells()
# masks.save("./output.tif")

my_mask = masks.getMasks().astype("int16")

pf, axarr = plt.subplots(1,1)
axarr.imshow(label2rgb(my_mask,bg_color=(0, 0, 0),bg_label=0))

#####