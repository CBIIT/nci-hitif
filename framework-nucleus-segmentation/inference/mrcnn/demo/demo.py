# Import all dependencies.
import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.insert(1, '../src')
import glob
import matplotlib.pyplot as plt
from skimage.color import label2rgb
from PIL import Image
from mrcnn_infer import *

# Setup Config File Path.
config_file_path='./demo_config.ini'

# Read Image files and preprocess
img_path = "./images/*.tif"
image_list = glob.glob(img_path)
img = np.zeros((len(image_list),1078,1278))
# mask = np.zeros((len(image_list),1078,1278))

for i in range(len(image_list)):
    img[i] = np.array(np.array(Image.open(image_list[i])))

# Plot Input images.
pf, axarr = plt.subplots(1,3)
axarr[0].imshow(img[0])
axarr[1].imshow(img[1])
axarr[2].imshow(img[2])
plt.rcParams['figure.figsize'] = [15, 15]

# Model and configuration file path.
mrcnn_model_path = "../model/mask_rcnn_cells_0194.h5"
config_file_path = "./demo_config.ini"

mask = mrcnn_infer(img, mrcnn_model_path, config_file_path)

pf, axarr = plt.subplots(1,3)
axarr[0].imshow(label2rgb(mask[0],bg_color=(0, 0, 0),bg_label=0))
axarr[1].imshow(label2rgb(mask[1],bg_color=(0, 0, 0),bg_label=0))
axarr[2].imshow(label2rgb(mask[2],bg_color=(0, 0, 0),bg_label=0))
plt.rcParams['figure.figsize'] = [15, 15]
