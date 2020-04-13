import os	
os.chdir("./framework-nucleus-segmentation/visualization/")

from download_util import *

GT_URL = 'https://ndownloader.figshare.com/files/22185189'
GS_URL = 'https://ndownloader.figshare.com/files/22185111'
INF_URL = 'https://ndownloader.figshare.com/files/22248840'

GT_PATH = 'GT.zip'
GS_PATH = 'Greyscale.zip'
INF_PATH = 'Inference.zip'

download_and_unzip_datasets(GT_URL, GT_PATH)
download_and_unzip_datasets(GS_URL, GS_PATH)