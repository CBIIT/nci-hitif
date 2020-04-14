import sys,os,configparser,skimage
import numpy as np
sys.path.insert(1, '../../../mrcnn/samples/cell')
from inference_utils import CleanMask
from inference import generate_inference_model
from inference import stitched_inference
from inference import map_uint16_to_uint8
from skimage.color import label2rgb

def mrcnn_infer(img, mrcnn_model_path, config_file_path):
    # Config File Parser
    config = configparser.ConfigParser()
    config.read(config_file_path)
    param = {s: dict(config.items(s)) for s in config.sections()}['general']
    for key in param:
        param[key] = int(param[key])
    cropsize = param['cropsize']
    padding = param['padding']
    threshold = param['threshold']

    model = generate_inference_model(mrcnn_model_path, cropsize)
    mask = np.zeros(img.shape)

    for i in range(len(img)):
        imarray = np.array(img[i]).astype('uint16')
        image = skimage.color.gray2rgb(imarray)
        image = map_uint16_to_uint8(image)

        stitched_inference_stack, num_times_visited = stitched_inference(image, cropsize, model, padding=padding)

        masks = CleanMask(stitched_inference_stack, threshold, )
        masks.merge_cells()

        my_mask = masks.getMasks().astype("int16")
        mask[i, :, :] = my_mask
    return mask
