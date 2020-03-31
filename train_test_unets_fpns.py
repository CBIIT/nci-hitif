from __future__ import print_function


import subprocess, re, os, sys, gc
from timeit import default_timer as timer

#### Load packages
import numpy as np
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, Dropout, core
from keras.optimizers import Adam, Adadelta, Nadam
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,EarlyStopping,CSVLogger
from keras import backend as K
from keras.models import load_model
from keras.layers.normalization import BatchNormalization
from keras.backend import binary_crossentropy
import keras
import random
import tensorflow as tf
from keras.models import model_from_json
import pandas as pd
import h5py

from segmentation_models import FPN
from segmentation_models import Unet
from segmentation_models.utils import set_trainable
from segmentation_models.backbones import get_feature_layers

from generator_h5_multioutput import DataGeneratorH5
import hitif_losses
from utils import Checkpoints
K.set_image_data_format('channels_last')  # TF dimension ordering in this code



def get_checkpoint_path(checkpoint_dir):
    """
    Return the path where the model checkpionts will be saved
    
    Argument:
        checkpoint_dir:
            The directory to save the models

    Returns:
        checkpoint_path: str
            The Keras path to save the model.
    """
   

    import datetime
    now = datetime.datetime.now()
    log_dir = os.path.join(checkpoint_dir, "{}{:%Y%m%dT%H%M}".format(
        model_name, now))


    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Path to save after each epoch. Include placeholders that get filled by Keras.
    checkpoint_path = os.path.join(log_dir, "weights_{}_*epoch*.h5".format(
        model_name))
    checkpoint_path = checkpoint_path.replace(
        "*epoch*", "{epoch:04d}")

    return checkpoint_path

def find_last(checkpoint_dir):
    """Finds the last checkpoint file of the last trained model in the
    model directory.
    Argument:
        checkpoint_dir:
            The directory to save the models

    Returns:
        The path of the last checkpoint file
    """
    # Get directory names. 

    checkpoint_dir = os.path.abspath(checkpoint_dir)
    dir_names = next(os.walk(checkpoint_dir))[1]
    key = model_name
    dir_names = filter(lambda f: f.startswith(key), dir_names)
    dir_names = sorted(dir_names)
    if not dir_names:
        import errno
        raise FileNotFoundError(
            errno.ENOENT,
            "Could not find model directory under {}".format(os.path.abspath(checkpoint_dir)))
    # Pick last directory
    dir_name = os.path.join(checkpoint_dir, dir_names[-1])
    # Find the last checkpoint
    checkpoints = next(os.walk(dir_name))[2]
    checkpoints = filter(lambda f: f.startswith("weights"), checkpoints)
    checkpoints = sorted(checkpoints)
    if not checkpoints:
        import errno
        raise FileNotFoundError(
            errno.ENOENT, "Could not find weight files in {}".format(dir_name))
    checkpoint = os.path.join(dir_name, checkpoints[-1])
    return checkpoint



def setup_one_gpu(gpu_id_idx=0):        
    #assert not 'tensorflow' in sys.modules, "GPU setup must happen before importing TensorFlow"
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  
    gpu_ids = str(os.environ["CUDA_VISIBLE_DEVICES"])
    print('Available GPUs = ', gpu_ids)
    gpu_ids_vec = gpu_ids.strip().split(',')
    # Pick the first available GPU
    gpu_id = gpu_ids_vec[gpu_id_idx]
    print("Picking GPU "+str(gpu_id))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

def setup_no_gpu():
    if 'tensorflow' in sys.modules:
        print("Warning, GPU setup must happen before importing TensorFlow")
    os.environ["CUDA_VISIBLE_DEVICES"] = ''

def loadpackages():
    import numpy as np
    from keras.models import Model
    from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, Dropout, core
    from keras.optimizers import Adam, Adadelta, Nadam
    from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,EarlyStopping,CSVLogger
    from keras import backend as K
    from keras.models import load_model
    from keras.layers.normalization import BatchNormalization
    from keras.backend import binary_crossentropy
    import keras
    import random
    import tensorflow as tf
    from keras.models import model_from_json
    import pandas as pd
    import h5py

    from segmentation_models import FPN
    from segmentation_models import Unet
    from segmentation_models.utils import set_trainable
    from segmentation_models.backbones import get_feature_layers

    from generator_h5_multioutput import DataGeneratorH5

    import hitif_losses
    K.set_image_data_format('channels_last')  # TF dimension ordering in this code


def get_model(model_json_fname, model_wts_fname):
    if os.path.isfile(model_json_fname):
         # Model reconstruction from JSON file
         with open(model_json_fname, 'r') as f:
            model = model_from_json(f.read())    
    #model.summary()     
    # Load weights into the new model
    weights = np.load(model_wts_fname, allow_pickle=True)
    model.set_weights(weights)
    #model.load_weights(model_wts_fname)
    return model      

def save_model_to_json(modeljsonfname):
    layers = get_feature_layers(backbone_name, 4)
    if backbone_type == 'FPN':
       model = FPN(input_shape=(None, None, num_channels), classes=num_mask_channels,encoder_weights=encoder_weights,backbone_name=backbone_name,activation=act_fcn,encoder_features=layers)
    elif backbone_type == 'Unet':
       model = Unet(input_shape=(None, None, num_channels), classes=num_mask_channels,encoder_weights=encoder_weights,backbone_name=backbone_name,activation=act_fcn, encoder_features=layers)
    #model.summary()
    # serialize model to JSON
    model_json = model.to_json()
    with open(modeljsonfname, "w") as json_file:
         json_file.write(model_json)

def train_generatorh5(params):
    from hitif_losses import dice_coef_loss_bce
    from hitif_losses import double_head_loss
	
    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)

    # Prepare for splitting the training set
    imgs_ind = np.arange(number_of_imgs)
    np.random.shuffle(imgs_ind)

    # Split 80-20
    train_last_id = int(number_of_imgs*0.80)
    
    # Generators
    training_generator = DataGeneratorH5(source_target_list_IDs=imgs_ind[0:train_last_id].copy(), **params)
    validation_generator = DataGeneratorH5(source_target_list_IDs=imgs_ind[train_last_id:number_of_imgs].copy(), **params)
   

    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    
    layers = get_feature_layers(backbone_name, 4)
    if backbone_type == 'FPN': 
       model = FPN(input_shape=(None, None, num_channels), classes=num_mask_channels,encoder_weights=encoder_weights,encoder_freeze=freezeFlag,backbone_name=backbone_name,activation=act_fcn, encoder_features=layers)
    elif backbone_type == 'Unet':
       model = Unet(input_shape=(None, None, num_channels), classes=num_mask_channels,encoder_weights=encoder_weights,encoder_freeze=freezeFlag,backbone_name=backbone_name,activation=act_fcn, encoder_features=layers)
    #model.summary()
    #model.compile(optimizer=Adam(lr=1e-5), loss='binary_crossentropy', metrics=['binary_crossentropy','mean_squared_error',dice_coef, dice_coef_batch, dice_coef_loss_bce,focal_loss()])
    #model.compile(optimizer=Adam(lr=1e-3), loss=dice_coef_loss_bce, metrics=['binary_crossentropy','mean_squared_error',dice_coef, dice_coef_batch,focal_loss()])
    #model.compile(optimizer=Adam(lr=1e-3), loss=loss_fcn, metrics=['binary_crossentropy','mean_squared_error',dice_coef, dice_coef_batch,focal_loss()])
    if loss_fcn == 'dice_coef_loss_bce':
        model.compile(optimizer=Adam(lr=1e-3), loss=dice_coef_loss_bce)
    elif loss_fcn == 'double_head_loss':
        model.compile(optimizer=Adam(lr=1e-3), loss=double_head_loss)
    else:
        model.compile(optimizer=Adam(lr=1e-3), loss=loss_fcn)

    # Loading previous weights for restarting
    if oldmodelwtsfname is not None:
       if os.path.isfile(oldmodelwtsfname) and reloadFlag :
          print('-'*30)
          print('Loading previous weights ...')

          weights = np.load(oldmodelwtsfname, allow_pickle=True)
          model.set_weights(weights)
          #model.load_weights(oldmodelwtsfname)

  
    checkpoint_path = get_checkpoint_path(log_dir_name)
    print("checkpoint_path:", checkpoint_path)
    model_checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, save_weights_only=True)
    custom_checkpoint = Checkpoints(checkpoint_path, monitor='val_loss', verbose = 1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,patience=25, min_lr=1e-6,verbose=1,restore_best_weights=True)
    model_es = EarlyStopping(monitor='val_loss', min_delta=1e-7, patience=15, verbose=1, mode='auto')
    csv_logger = CSVLogger(csvfname, append=True)

    #my_callbacks = [reduce_lr, model_es, csv_logger]
    #my_callbacks = [model_checkpoint, reduce_lr, model_es, csv_logger]
    my_callbacks = [custom_checkpoint, reduce_lr, model_es, csv_logger]
    print('-'*30)
    print('Fitting model encoder freeze...')
    print('-'*30)
    if freezeFlag and num_coldstart_epochs > 0:
        model.fit_generator(generator=training_generator,validation_data=validation_generator,use_multiprocessing=True,workers=num_gen_workers,epochs=num_coldstart_epochs,callbacks= my_callbacks, verbose=2)

    # release all layers for training
    set_trainable(model) # set all layers trainable and recompile model
    #model.summary()

    print('-'*30)
    print('Fitting full model...')
    print('-'*30)

    ## Retrain after the cold-start
    model.fit_generator(generator=training_generator,validation_data=validation_generator,use_multiprocessing=True,workers=num_gen_workers,epochs=num_finetuning_epochs,callbacks=my_callbacks, verbose=2)

    ## <<FIXME>>: GZ will work on it.
    # Find the last best epoch model weights and then symlink it to the modelwtsfname
    # Note that the symlink will have issues on NON-Linux OS so it is better to copy.
    '''
    os.symlink(<last_best_h5>, modelwtsfname)
    # Or copy last best weights
    last_best_modelwts_fname = find_last(log_dir_name)
    from shutil import copyfile
    copyfile(last_best_modelwts_fname, modelwtsnfname)
    '''
    # end

def rotateandflipimages(img):
    assert(img.shape[0] == img.shape[1])
    out_imgs = np.zeros([6, img.shape[0], img.shape[1]], dtype='float32')
    out_imgs[1,:,:] = np.fliplr(img)
    out_imgs[2,:,:] = np.flipud(img)
    out_imgs[3,:,:] = np.rot90(img, k = 1)
    out_imgs[4,:,:] = np.rot90(img, k = 2)
    out_imgs[5,:,:] = np.rot90(img, k = -1)
    out_imgs[0,:,:] = img
    #print('done')
    return out_imgs

def reverserotateandflipimages(imgs):
    assert(imgs.shape[1] == imgs.shape[2])
    if imgs.ndim == 3:
       out_img = np.zeros([imgs.shape[1], imgs.shape[2]], dtype='float32')
       tmp_imgs = np.zeros([6, imgs.shape[1], imgs.shape[2]], dtype='float32')
       tmp_imgs[1,:,:] = np.fliplr(np.squeeze(imgs[1,:,:]))
       tmp_imgs[2,:,:] = np.flipud(np.squeeze(imgs[2,:,:]))
       tmp_imgs[3,:,:] = np.rot90(np.squeeze(imgs[3,:,:]), k = -1)
       tmp_imgs[4,:,:] = np.rot90(np.squeeze(imgs[4,:,:]), k = -2)
       tmp_imgs[5,:,:] = np.rot90(np.squeeze(imgs[5,:,:]), k = 1)
       tmp_imgs[0,:,:] = np.squeeze(imgs[0,:,:])
       out_img = np.squeeze(np.mean(tmp_imgs, axis=0))
       #print("Min/Max = ", np.amin(out_img), np.amax(out_img))
    if imgs.ndim == 4:
       out_img = np.zeros([imgs.shape[1], imgs.shape[2], imgs.shape[3]], dtype='float32')
       for j in range(imgs.shape[3]):
           tmp_imgs = np.zeros([6, imgs.shape[1], imgs.shape[2]], dtype='float32')
           tmp_imgs[1,:,:] = np.fliplr(np.squeeze(imgs[1,:,:,j]))
           tmp_imgs[2,:,:] = np.flipud(np.squeeze(imgs[2,:,:,j]))
           tmp_imgs[3,:,:] = np.rot90(np.squeeze(imgs[3,:,:,j]), k = -1)
           tmp_imgs[4,:,:] = np.rot90(np.squeeze(imgs[4,:,:,j]), k = -2)
           tmp_imgs[5,:,:] = np.rot90(np.squeeze(imgs[5,:,:,j]), k = 1)
           tmp_imgs[0,:,:] = np.squeeze(imgs[0,:,:,j])
           out_img[:,:,j] = np.squeeze(np.mean(tmp_imgs, axis=0))
           #print("Min/Max = ", np.amin(out_img), np.amax(out_img))

    return out_img, tmp_imgs

def predict(modeljsonfname, modelweightsfname):

    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)
    # 
    imgs_train = np.load(testinputnpyfname).astype('float32') 

    print('Shape of input images ', imgs_train.shape)    
    print('Input images min, max before normalization ', np.amin(imgs_train),np.amax(imgs_train))

    # Scale Input Greyscale to [0,1]
    imgs_train = imgs_train/255.0
    print('Input images min, max after scaling ', np.amin(imgs_train),np.amax(imgs_train))

    print('-'*30)
    print('Loading model...')
    print('-'*30)

    model = get_model(modeljsonfname, modelweightsfname)

    print('-'*30)
    print('Predicting model...')
    print('-'*30)

    if expandChannel:
       imgs_train = np.stack((imgs_train,)*3, -1)
    else:
       imgs_train = np.expand_dims(imgs_train, num_channels)

    print('Test Images shape =', imgs_train.shape)

    imgs_test_predict = model.predict(imgs_train, batch_size=infer_batch_size, verbose=1)
    print('Target prediction min, max ', np.amin(imgs_test_predict),np.amax(imgs_test_predict))
    print('Shape of Output images ', imgs_test_predict.shape)

    np.save(testpredictnpyfname, np.squeeze(imgs_test_predict))
    #
    del model

def predicttta(modeljsonfname, modelweightsfname):

    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)
    # 
    imgs_train = np.load(testinputnpyfname).astype('float32') 

    print('Shape of input images ', imgs_train.shape)    
    print('Input images min, max before normalization ', np.amin(imgs_train),np.amax(imgs_train))

    # Scale Input Greyscale to [0,1]
    imgs_train = imgs_train/255.0
    print('Input images min, max after scaling ', np.amin(imgs_train),np.amax(imgs_train))
    ## Expand inputs
    totimgs = imgs_train.shape[0]
    imgs_train_n = np.zeros([totimgs*6, imgs_train.shape[1], imgs_train.shape[2]], dtype='float32')
    for i in range(totimgs):
        im = np.squeeze(imgs_train[i,:,:])
        im_rf = rotateandflipimages(im)
        imgs_train_n[(i*6):(i+1)*6,:,:] = im_rf
        #print('Done with ', i, ' of ', totimgs)
 
    print('Shape of input images after augmentation ', imgs_train_n.shape)    

    print('-'*30)
    print('Loading model...')
    print('-'*30)

    model = get_model(modeljsonfname, modelweightsfname)

    print('-'*30)
    print('Predicting model...')
    print('-'*30)

    if expandChannel:
       imgs_train_n = np.stack((imgs_train_n,)*3, -1)
    else:
       imgs_train_n = np.expand_dims(imgs_train_n, num_channels)

    print('Test Images shape =', imgs_train_n.shape)

    imgs_test_predict = model.predict(imgs_train_n, batch_size=infer_batch_size, verbose=1)
    print('Target prediction min, max ', np.amin(imgs_test_predict),np.amax(imgs_test_predict))
    print('Shape of Output images before Compressing Augmentation ', imgs_test_predict.shape)
    ## Shrink output
    if imgs_test_predict.shape[3] == 1:
       imgs_test_predict_n = np.zeros([totimgs, imgs_test_predict.shape[1], imgs_test_predict.shape[2]], dtype='float32')
    elif imgs_test_predict.shape[3] > 1:
       imgs_test_predict_n = np.zeros([totimgs, imgs_test_predict.shape[1], imgs_test_predict.shape[2], imgs_test_predict.shape[3]], dtype='float32')
    for i in range(totimgs):
        for j in range(imgs_test_predict.shape[3]):
            imgs = np.squeeze(imgs_test_predict[(i*6):(i+1)*6,:,:,j])
            im_rf, _ = reverserotateandflipimages(imgs)
            if imgs_test_predict.shape[3] == 1:
               imgs_test_predict_n[i,:,:] = im_rf
            if imgs_test_predict.shape[3] > 1:
               imgs_test_predict_n[i,:,:,j] = im_rf
            #del imgs_test_predict
    print('Shape of Output images after Compressing Augmentation ', imgs_test_predict_n.shape)    
    np.save(testpredictnpyfname, imgs_test_predict_n)
    #
    del model

def geth5datasetshape(h5fname, keys):
    with h5py.File(h5fname, "r") as hf:
        for key in list(keys):
            dset = hf[key]
            dset_shape = np.array(dset.shape).tolist()
            dset_type = dset.dtype
            print('Selected Target Dataset name = ', key, ' Shape = ', dset_shape, ' DTYPE = ', dset_type)
    return dset_shape, dset_type


def print_args():
    print("-----------------------------------------------------")
    print("-----------------------------------------------------")
    print("h5fname = ", h5fname)
    print("source_dataset_name = ", src_datasetname)
    print("target_dataset_name = ", tar_datasetname)
    print("-----------------------------------------------------")
    print("-----------------------------------------------------")
    print("total_number_of_training_images = ", number_of_imgs)
    print("input_image_height = ", img_rows)
    print("input_image_width = ", img_cols)
    print("source_channels = ", num_channels)
    print("target_channels = ", num_mask_channels)
    print("target_abs_value_Flag = ", target_abs_value_Flag)
    print("-----------------------------------------------------")
    print("-----------------------------------------------------")
    print("architecture_type = ", backbone_type)
    print("backbone_name = ", backbone_name)
    print("backbone_encoder_weights = ", encoder_weights)
    print("backbone_encoder_freeze_flag = ", freezeFlag)
    print("model_weights_reload_flag = ", reloadFlag)
    print("-----------------------------------------------------")
    print("-----------------------------------------------------")
    print("training batch size = ", train_batch_size)
    print("number of cold-start epochs = ", num_coldstart_epochs)
    print("number of fine-tuning epochs = ", num_finetuning_epochs)
    print("useTTA For Prediction = ", predictTTAFlag)
    print("last_layer_activation_function = ", act_fcn)
    print("loss_function = ", str(loss_fcn))
    print("-----------------------------------------------------")
    print("-----------------------------------------------------")
    print("old_model_wts_fname = ", oldmodelwtsfname)
    print("model_wts_fname = ", modelwtsfname)
    print("model_json_fname = ", model_json_fname)
    print("log_file_fname = ", csvfname)
    print("-----------------------------------------------------")
    print("-----------------------------------------------------")

if __name__ == '__main__':
    
    from params import args

    # Required, Index for selecting GPU
    gpu_id_idx = args.gpuPCIID
    #gpu_id_idx = 0

    # Pick a GPU
    setup_one_gpu(gpu_id_idx)


    #set up log dir 
    checkpoint_dir =  "models"
    print("-----------------------------------------------------")
    print("-----------------------------------------------------")  
   	
    # Optional, default source channels
    num_channels = 1

    # Required, copy single channel to 3 channels.
    # This should be true if you UResNets/FPNs as they expect input to be (None, None, 3). This is without batchsize.
    expandChannel = args.expandInputChannels
    
    if expandChannel:
       num_channels = 3

    # Required, number of target channels
    num_mask_channels = args.numberofTargetChannels
    # For h5 generator constructor
    n_in_channels = num_channels
    n_out_channels = num_mask_channels

    ##
    # Required backbone weights
    freezeFlag = args.freezeEncoderBackbone
    # Required, use existing weights
    reloadFlag = args.reloadExistingWeights

    # Optional,
    oldmodelwtsfname = args.oldWeightsFilename
    ## All are required
    '''
    backbone_type = "FPN" # "FPN" or "Unet"
    backbone_name =  "resnet152" # "inceptionresnetv2", "resnet152"
    encoder_weights =  "imagenet11k" # "imagenet", "None", "imagenet11k"
    act_fcn = "sigmoid" # "linear", "sigmoid", "softmax", "tanh", etc.
    prefix_model= "ResNet152FPN" # "IncResV2FPN", "ResNet152FPN", UIncResV2Net, UResNet152Net
    suffix_model="input_normalizeandscaleUint8_target_mask_normalizeMax_bcedice"
    testdata_prefix = "ResNet152FPN" # "IncResV2FPN", "ResNet152FPN", UIncResV2Net, UResNet152Net
    '''
    backbone_type = args.architectureType # "FPN" or "Unet"
    backbone_name =  args.backboneName # "inceptionresnetv2", "resnet152"
    encoder_weights =  args.backboneEncoderWeightsName # "imagenet", "None", "imagenet11k"
    if encoder_weights == "None":
        encoder_weights = None
    act_fcn = args.activationFunction # "linear", "sigmoid" "softmax", "tanh", etc.
    prefix_model= args.outputModelPrefix # "IncResV2FPN", "ResNet152FPN", UIncResV2Net, UResNet152Net
    suffix_model=args.outputModelSuffix
    testdata_prefix = args.outputModelPrefix  # "IncResV2FPN", "ResNet152FPN", UIncResV2Net, UResNet152Net

    # Required, Objective/Loss function
    loss_fcn = args.loss_function # 'mean_squared_error', 'mean_absolute_error', 'binary_crossentropy', 'dice_coef_loss_bce'

    # Optional, Model (weights ONLY), architecture and training history
    #modelwtsfname = './models/'+prefix_model+'_weightsONLY_'+suffix_model+'.h5'
    #model_json_fname  = './models/'+prefix_model+'_modelarch_'+suffix_model+'.json'

    modelwtsfname = args.trained_h5 
    model_json_fname  = args.trained_json 
    history_dir = "history"
    if not os.path.exists("history"):
        os.mkdir(history_dir)
    csvfname = os.path.join(history_dir, prefix_model+'_history_'+suffix_model+'.csv')

    # Required, batch sizes for training and prediction
    train_batch_size = args.trainingBatchSize
    infer_batch_size = args.inferenceBatchSize

    ## Required, h5fname containing the source and targets
    h5fname = args.h5fname
    
    ## Typical h5 dataset(s) generated by the KNIME Workflow
    ## knime://knime-server/HiTIF/Reddy/DL_Segmentation_Paper/HiTIF_AugmentInputGT_H5
    '''
    Dataset Name =  DAPI_uint16touint8_normalizeandscale  Shape =  [10500, 256, 256]  Type =  uint8
    Dataset Name =  alongboundary_bool  Shape =  [10500, 256, 256]  Type =  bool
    Dataset Name =  bitmask_bool  Shape =  [10500, 256, 256]  Type =  bool
    Dataset Name =  bitmask_erosion_bool  Shape =  [10500, 256, 256]  Type =  bool
    Dataset Name =  bitmask_labeled_uint16  Shape =  [10500, 256, 256]  Type =  uint16
    Dataset Name =  bkg_bool  Shape =  [10500, 256, 256]  Type =  bool
    Dataset Name =  distancemapnormalized_float32  Shape =  [10500, 256, 256]  Type =  float32
    Dataset Name =  gblurradius1_float32  Shape =  [10500, 256, 256]  Type =  float32
    '''
    # Required
    src_datasetname = [args.srcDatasetName]
    # Required# or multiple targets of same DTYPE ['bitmask_bool', 'outsideboundary_bool']
    tar_datasetname = args.tarDatasetName
    # Required
    target_abs_value_Flag = args.targetDontUseAbsoluteValue # True only for gblurradius1_float32 target datasetname
    
    # Get Source and Target Dataset Shape and Type
    print("-----------------------------------------------------")
    print("-----------------------------------------------------")  
    src_dset_shape, src_dset_type = geth5datasetshape(h5fname, src_datasetname)
    tar_dset_shape, tar_dset_type = geth5datasetshape(h5fname, tar_datasetname)
    tar_datasetname = list(tar_datasetname)
    
    number_of_imgs = src_dset_shape[0]
    img_rows = src_dset_shape[1]
    img_cols = src_dset_shape[2]

    scale_source_Flag = False
    scale_target_Flag = False # Default for 'float32' type data

    if src_dset_type == "uint8":
       scale_source_Flag = True
    if tar_dset_type is not "float32":
       scale_target_Flag = True
    
    # Generator Parameters
    h5generator_params = {'dim': (img_rows,img_cols),
          'h5fname': h5fname, 
          'source_datasetname': src_datasetname,
          'target_datasetname': tar_datasetname,          
          'batch_size': train_batch_size,
          'n_in_channels': num_channels,
          'n_out_channels': num_mask_channels,
          'scale_source': scale_source_Flag,
          'scale_target': scale_target_Flag,
          'target_abs_value': target_abs_value_Flag,
          'shuffle': True}
    # Set number of generator workers the train_batch_size
    # Does this need to be greater than number train_batch_size???  
    num_gen_workers = train_batch_size
    # Set number of fine-tuning epochs
    num_finetuning_epochs = args.numFTEpochs
    # Set number of cold start epochs
    num_coldstart_epochs = args.numCSEpochs
    ##Optional, For Prediction
    predictTTAFlag = args.useTTAForTesting

    #Required, Uint8 NUMPY
    testinputnpyfname = args.testinputnumpyfname
    #Required, Prediction NUMPY
    testpredictnpyfname = args.testpredictnumpyfname


    model_name = "unet_fpn" 
    slurm_job_id = os.environ['SLURM_JOB_ID']
    #log_dir_name = os.path.join("/lscratch", slurm_job_id, "logs")
    log_dir_name = os.path.join(os.path.abspath("."))

    ##print the arguments
    print_args()
  
    mode = args.mode
    # mode ['generate', 'train', 'infer']
    ## Main functions
    if mode == "generate":
       save_model_to_json(model_json_fname)
    elif mode == "train":
       save_model_to_json(model_json_fname)
       train_generatorh5(h5generator_params)
       best_model = os.path.abspath(find_last(log_dir_name))
       from shutil import copyfile
       os.symlink(best_model, modelwtsfname)
       print('\n Best model {0} symlinked to {1}'.format(best_model, modelwtsfname))
       exit(0)
       #copyfile(best_model,modelwtsfname)
       #print('\n Last model {0} saved'.format(modelwtsfname))
    elif mode == "infer":
       if predictTTAFlag:
          predicttta(model_json_fname, modelwtsfname)
       else:
          predict(model_json_fname, modelwtsfname)   
