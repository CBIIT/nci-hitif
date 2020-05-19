from skimage.util import img_as_ubyte
import line_profiler
from line_profiler import LineProfiler
import sys,glob,warnings,os
import matplotlib.pyplot as plt
from PIL import Image
from skimage.transform import resize
from skimage.color import label2rgb

sys.path.insert(1, '../src');
sys.path.insert(1,'../../../visualization')
from mrcnn_infer_profile import *
from download_util import *

MRCNN_MODEL_URL = 'https://ndownloader.figshare.com/files/22280580?private_link=dd27a1ea28ce434aa7d4'
MRCNN_MODEL_PATH = 'MRCNN_pretrained.zip'
download_and_unzip_datasets(MRCNN_MODEL_URL, MRCNN_MODEL_PATH)
mrcnn_model_path = "./mrcnn_pretrained.h5"

config_file_path = "./demo.ini"
with open(config_file_path, 'r') as fin:
    print(fin.read())

image_list =['../../../visualization/GreyScale/BABE_Biological/Plate1_E03_T0001FF001Zall.tif',
             '../../../visualization/GreyScale/HiTIF_Laurent_Technical/AUTO0496_J14_T0001F001L01A01Z01C01.tif',
             '../../../visualization/GreyScale/Manasi_Technical/Plate1_M21_T0001F003L01A01Z01C01.tif'
]

img = np.zeros((1,1078,1278))
image_resized = img_as_ubyte(resize(np.array(Image.open(image_list[0])), (1078, 1278)))
for i in range(len(img)):
    img[i,:,:] = image_resized

mask = mrcnn_infer(img, mrcnn_model_path, config_file_path)
lp = LineProfiler()
lp_wrapper = lp(mrcnn_infer)
lp_wrapper(img, mrcnn_model_path, config_file_path)
lp.print_stats()



