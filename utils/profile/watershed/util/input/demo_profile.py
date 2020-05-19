import sys,glob
import matplotlib.pyplot as plt
from PIL import Image
from skimage.transform import resize
from skimage.util import img_as_ubyte
from skimage.color import label2rgb
from line_profiler import LineProfiler

sys.path.insert(1, '../src');
sys.path.insert(1,'../../../visualization')
from watershed_infer_profile import *
from download_util import *

BLUR_MODEL_URL = 'https://ndownloader.figshare.com/files/22280349?private_link=a3fec498ef6d08ac6973'
BLUR_MODEL_PATH = 'blurred_border_FPN_pretrained.zip'
download_and_unzip_datasets(BLUR_MODEL_URL, BLUR_MODEL_PATH)

modelwtsfname = "./blurred_border_FPN_pretrained.npy"
modeljsonfname = "./blurred_border_FPN_pretrained.json"
gaussian_blur_model = get_model(modeljsonfname,modelwtsfname)

DISTANCE_MAP_MODEL_URL = 'https://ndownloader.figshare.com/files/22280352?private_link=5b1454e3f3bd23dea56f'
DISTANCE_MAP_MODEL_PATH = 'distance_map_FPN_pretrained.zip'
download_and_unzip_datasets(DISTANCE_MAP_MODEL_URL, DISTANCE_MAP_MODEL_PATH)

modelwtsfname = "./distance_map_FPN_pretrained.npy"
modeljsonfname = "./distance_map_FPN_pretrained.json"
distance_map_model = get_model(modeljsonfname,modelwtsfname)

config_file_path='./demo.ini'
with open(config_file_path, 'r') as fin:
    print(fin.read())

image_list =['../../../visualization/GreyScale/BABE_Biological/Plate1_E03_T0001FF001Zall.tif',
             '../../../visualization/GreyScale/HiTIF_Laurent_Technical/AUTO0496_J14_T0001F001L01A01Z01C01.tif',
             '../../../visualization/GreyScale/Manasi_Technical/Plate1_M21_T0001F003L01A01Z01C01.tif'
]

img = np.zeros(($ITER,1078,1278))
image_resized = img_as_ubyte(resize(np.array(Image.open(image_list[0])), (1078, 1278)))
for i in range(len(img)):
    img[i,:,:] = image_resized

mask = watershed_infer(img,gaussian_blur_model,distance_map_model,config_file_path)
