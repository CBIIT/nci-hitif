import requests
import zipfile
import glob
import numpy as np
from PIL import Image

def get_mrcnn_model():
	_URL = 'https://ndownloader.figshare.com/files/22236213?private_link=dd27a1ea28ce434aa7d4'
	open('mask_rcnn_cells_0194.h5', 'wb').write((requests.get(_URL, allow_redirects=True)).content)

def get_sample_images():
	_URL = "https://ndownloader.figshare.com/articles/12107529?private_link=1b82745b8a3dc89a3a86"
	open('images.zip', 'wb').write((requests.get(_URL, allow_redirects=True)).content)
	with zipfile.ZipFile('images.zip', 'r') as zip_ref:
		zip_ref.extractall('./images')

	img_path = "./images/*.tif"
	image_list = glob.glob(img_path)
	img = np.zeros((len(image_list),1078,1278))

	for i in range(len(image_list)):
	    img[i,:,:] = np.array(Image.open(image_list[i]))

	return img

