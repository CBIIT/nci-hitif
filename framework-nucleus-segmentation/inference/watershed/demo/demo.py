# Import all dependencies.
import warnings
warnings.filterwarnings("ignore")
import sys
from skimage.color import label2rgb
import glob
import matplotlib.pyplot as plt

# Import main python module (watershed_infer).
sys.path.insert(1, '../src')
from watershed_infer import *

# Setup Config File Path.
config_file_path='./demo_config.ini'

# Read Images.
img_path = "./images/*.tif"
image_list = glob.glob(img_path)
img = np.zeros((len(image_list),1078,1278))

for i in range(len(image_list)):
    img[i,:,:] = cv2.imread(image_list[i], -1)

# Plot Input images.
pf, axarr = plt.subplots(1,3)
axarr[0].imshow(img[0])
axarr[1].imshow(img[1])
axarr[2].imshow(img[2])
plt.rcParams['figure.figsize'] = [15, 15]

# Load Pre-trained model (Gaussian Blur DL Model).
modelwtsfname = "../model/run010/gaussian/trained.npy"
modeljsonfname = "../model/run010/gaussian/trained.json"
gaussian_blur_model = get_model(modeljsonfname,modelwtsfname)

# Load Pre-trained model (Distance Map DL Model).
modelwtsfname = "../model/run010/edt/trained.npy"
modeljsonfname = "../model/run010/edt/trained.json"
distance_map_model = get_model(modeljsonfname,modelwtsfname)

# Prediction (Instance Segmentation).
mask = watershed_infer(img,gaussian_blur_model,distance_map_model,config_file_path)

# Plot the result.
pf, axarr = plt.subplots(1,3)
axarr[0].imshow(label2rgb(mask[0],bg_color=(0, 0, 0),bg_label=0))
axarr[1].imshow(label2rgb(mask[1],bg_color=(0, 0, 0),bg_label=0))
axarr[2].imshow(label2rgb(mask[2],bg_color=(0, 0, 0),bg_label=0))
plt.rcParams['figure.figsize'] = [15, 15]

# plt.imshow(label2rgb(mask[0], (img[0]/256).astype('uint8'),alpha=0.1,image_alpha=1,bg_color=(0, 0, 0),bg_label=0))
