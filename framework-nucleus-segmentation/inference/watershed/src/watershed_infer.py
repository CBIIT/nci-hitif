from remove_large_objects import *
from skimage.morphology import remove_small_objects
from skimage.segmentation import watershed
from skimage.measure import label
from skimage.morphology import closing, disk
from skimage.transform import resize
import configparser
# import tensorflow as tf
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.optimizers import Adam
from keras import backend as K
from segmentation_models.common.layers import ResizeImage
import os
import numpy as np
import cv2
from keras.models import model_from_json

# Function for reading pre-trained model.
def get_model(modeljsonfname,modelwtsfname):
    with open(modeljsonfname, 'r') as f:
        model = model_from_json(f.read())

    # Load weights into the new model
    if modelwtsfname.endswith(".h5"):
        model.load_weights(modelwtsfname)
    elif modelwtsfname.endswith(".npy"):
        weights = np.load(modelwtsfname, allow_pickle=True)
        model.set_weights(weights)
    else:
        print("ERROR: model weights in unknown format: {0}".format(modelwtsfname))
        exit(1)
    return model

# Model Prediction function
def unet_predict(model, batch_size, imgs_test):
    model_input_channelSize = model.layers[0].input_shape[-1]
    imgs_test = imgs_test.astype('float32')

    if model_input_channelSize > 1:
        imgs_test = np.stack((imgs_test,) * model_input_channelSize, -1)
    elif model_input_channelSize == 1:
        imgs_test = np.expand_dims(imgs_test, 3)
    imgs_mask_test = model.predict(imgs_test, batch_size=batch_size, verbose=1)

    return (imgs_mask_test)


def model_prediction(img,model,param):

    # Change the datatype with normalization. (u16 -> ubyte)
    img = cv2.normalize(src=img, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Change the input resolution into padded resolution.
    dim1 = param['padded_width']
    dim2 = param['padded_height']
    dim_original_height, dim_original_width = img[0].shape

    imshape = np.array([dim2, dim1]).astype('uint64')
    noofImages = img.shape[0]

    batch_size = param['batch_size']
    imagesNP = np.zeros([noofImages, imshape[0], imshape[1]], dtype=np.float32)

    for index in range(len(img)):
        input_cell = img[index, :, :]
        im_in = input_cell.astype('float32')
        c_im_in_shape = np.array(im_in.shape)
        c_im_in_shape_pad = ((c_im_in_shape - imshape) / 2).astype('int')
        im_in_pad = np.lib.pad(im_in, (
            (-c_im_in_shape_pad[0], -c_im_in_shape_pad[0]), (-c_im_in_shape_pad[1], -c_im_in_shape_pad[1])),
                               'constant').copy()
        del im_in
        im_in = im_in_pad
        del im_in_pad

        xs = 0
        ys = 0
        xe = im_in.shape[0]
        ye = im_in.shape[1]

        c_im_in_max = np.amax(im_in)
        if c_im_in_max > 255:
            imagesNP[index, xs:xe, ys:ye] = im_in / float((2 ** 16) - 1)
        elif 0 <= c_im_in_max <= 255:
            imagesNP[index, xs:xe, ys:ye] = im_in / float((2 ** 8) - 1)
        elif 0 <= c_im_in_max <= 1.0:
            imagesNP[index, xs:xe, ys:ye] = im_in

    imgs_mask = unet_predict(model, batch_size, imagesNP)

    result = np.zeros((noofImages, dim_original_height, dim_original_width))

    for i in range(noofImages):
        yoffset = int((dim2 - dim_original_height) / 2)
        xoffset = int((dim1 - dim_original_width) / 2)
        if imgs_mask.ndim == 4:
            im_in = (imgs_mask[i, :, :, :])
            im_in = im_in[yoffset:yoffset + dim_original_height,
                    xoffset:xoffset + dim_original_width, :]
            im_in = np.swapaxes(im_in, 0, 2)
            im_in = np.transpose(im_in, (0, 2, 1))
            if im_in.shape[0] == 1:
                im_in = np.squeeze(im_in)
        if imgs_mask.ndim == 3:
            im_in = np.squeeze(imgs_mask[i, :, :])
            im_in = im_in[yoffset:yoffset + dim_original_height,
                    xoffset:xoffset + dim_original_width]
        if imgs_mask.ndim == 2:
            im_in = imgs_mask
            im_in = im_in[yoffset:yoffset + dim_original_height,
                    xoffset:xoffset + dim_original_width]
        result[i,:,:] = im_in

    return result

def watershed_infer(img,gaussian_blur_model,distance_map_model,config_file_path):

    # Parser (Config file -> Dictionary)
    config = configparser.ConfigParser()
    config.read(config_file_path)
    param = {s: dict(config.items(s)) for s in config.sections()}['general']

    for key in param:
        if key == "global_threshold":
            param[key] = float(param[key])
        else:
            param[key] = int(param[key])

    # Prediciton (Galussian_Blur DL model)
    img_gauss = model_prediction(img, gaussian_blur_model, param)

    # Prediction (Distance MAP DL Model)
    result = model_prediction(img_gauss, distance_map_model, param)

    # Global Thresholding
    result[result<param['global_threshold']] = 0
    result[result!=0] = 1

    mask = np.zeros(result.shape)

    for i in range(len(result)):

        # Morphorligcal Operation
        mask[i] = closing(result[i], disk(6))

        # Connected Component Analysis
        mask[i] = label(mask[i], connectivity=1, background=0)

        # Watershed
        watershed_result = watershed(img_gauss[i], mask[i])

        # Label size filter
        mask[i] = remove_small_objects(watershed_result, param['label_min_size'], connectivity=1)
        mask[i] = remove_large_objects(watershed_result, param['label_max_size'], connectivity=1)

    return mask
