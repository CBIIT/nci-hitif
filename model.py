from __future__ import print_function

import os
import numpy as np
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,EarlyStopping, CSVLogger
from keras import backend as K
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import argparse
import pickle


K.set_image_data_format('channels_last')  # TF dimension ordering in this code
__file__="/gpfs/gsfs2/users/zakigf/swift-t/examples/nci-hitif/model.py"
#img_rows = 128
#img_cols = 128

smooth = 1.

reloadFlag = True
savePNGS = False
#oldmodelwtsfname = 'short_weights_nodil_normalize.h5'
modelwtsfname = 'model_weights.h5'


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)



def get_unet (img_rows, img_cols, n_layers=5, filter_size=32, dropout=None, activation_func="relu", conv_size=3 ):

    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    print (img_rows)
    print (img_cols)
    inputs = Input((img_rows, img_cols, 1))
    conv_layers=[] 
    pool_layers=[inputs]
    conv_filter=(conv_size, conv_size )


    for i in range(n_layers):
        conv = Conv2D(filter_size,  conv_filter, activation=activation_func, padding='same')(pool_layers[i])
        if dropout != None:
            conv = Dropout(dropout)(conv)
        conv = Conv2D(filter_size, conv_filter, activation=activation_func, padding='same')(conv)
        pool = MaxPooling2D(pool_size=(2, 2))(conv)
        conv_layers.append(conv)
        pool_layers.append(pool)
        filter_size *=2
        
  
    filter_size /=4

    for i in range(n_layers-1):
        up = concatenate([Conv2DTranspose(filter_size, (2, 2), strides=(2, 2), padding='same')(conv_layers[-1]), conv_layers[n_layers-i-2]], axis=3)
        conv = Conv2D(filter_size, conv_filter, activation=activation_func, padding='same')(up)
        if dropout != None:  
            conv = Dropout(dropout)(conv)
        conv = Conv2D(filter_size, conv_filter, activation=activation_func, padding='same')(conv)
        conv_layers.append(conv)
        filter_size /= 2


    last_conv =  Conv2D(1, (1, 1), activation='sigmoid')(conv_layers[-1])
    conv_layers.append(last_conv)

    model = Model(inputs=[inputs], outputs=[last_conv])


    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])
    model.summary()

    return model


def get_images(images, masks, augmentation_factor=None, normalize_mask=False):

    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)

    #imgs_train_file="1CDT_Green_Red_FarRed_Annotated_FISH_Dilation4Conn1Iter_Training_128by128_normalize.npy"
    #imgs_mask_file="1CDT_Green_Red_FarRed_Annotated_FISH_Dilation4Conn1Iter_Training_128by128_normalize_Mask.npy"
    #file_path = os.path.dirname(os.path.realpath(__file__))
    #imgs_train_path=os.path.join(file_path, "data_python", imgs_train_file)
    #imgs_mask_path=os.path.join(file_path, "data_python", imgs_mask_file)


    imgs_train = np.squeeze(np.load(images))
    imgs_mask_train = np.squeeze(np.load(masks))


    if imgs_train.ndim != 3:
        raise Exception("Error: The number of dimensions for images should equal 3, after squeezing the shape is:{0}".format(np.shape(images)))

    if imgs_mask_train.ndim != 3:
        raise Exception("Error: The number of dimensions for masks should equal 3, after squeezing the shape is:{0}".format(np.shape(masks)))

    imgs_train = imgs_train.astype('float32')
    imgs_mask_train = imgs_mask_train.astype('float32')
    if normalize_mask:
        imgs_mask_train /= 255.  # scale masks to [0, 1]


    imgs_train = np.expand_dims(imgs_train, axis= 3)
    imgs_mask_train = np.expand_dims(imgs_mask_train, axis= 3)

    return_images = imgs_train 
    return_masks = imgs_mask_train 
     
    if augmentation_factor != None:
    
        sample_size = len(imgs_train) 
        augmented_sample_size = int(float(sample_size) * augmentation_factor)
        print ("Expanding the images from:{0} to {1}\n".format(sample_size, augmented_sample_size))
    
        #data_gen_args = dict( rotation_range=90.,
        data_gen_args = dict( width_shift_range=0.1,
                height_shift_range=0.1,
                horizontal_flip=True,
                vertical_flip=True,
                )
        image_datagen = ImageDataGenerator(**data_gen_args)
        mask_datagen = ImageDataGenerator(**data_gen_args)
    
        augmented_images = []
        augmented_masks = []
    
        i = 0     
        for batch in image_datagen.flow(imgs_train, batch_size=1, seed=1337): 
            augmented_images.append(np.squeeze(batch, axis=0))
            i = i + 1 
            if i >= augmented_sample_size:
                break
    
        i = 0     
        for batch in image_datagen.flow(imgs_mask_train, batch_size=1, seed=1337): 
            augmented_masks.append(np.squeeze(batch,axis=0))
            i = i +1
            if i >=  augmented_sample_size:
                break

        return_images = np.stack(augmented_images, axis=0)
        return_masks = np.stack(augmented_masks, axis=0)

    print (np.shape(return_images))
    print (np.shape(return_masks))
    return [return_images, return_masks]

def train(model, imgs_train, imgs_mask_train, initialize=None):

    model_checkpoint = ModelCheckpoint(modelwtsfname, monitor='val_loss', save_best_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,patience=50, min_lr=0.001,verbose=1)
    model_es = EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=50, verbose=1, mode='auto')
    csv_logger = CSVLogger('training.csv')

    print('-'*30)
    print('Fitting model...')
    print('-'*30)

    if initialize != None:
        print("Initializing the model using:{0}\n", initialize)
        model.load_weights(initialize)
        
    print (np.shape(imgs_train))
    print (np.shape(imgs_mask_train))
    #return model.fit(imgs_train, imgs_mask_train, batch_size=2, epochs=3000, verbose=2, shuffle=True,
    return model.fit(imgs_train, imgs_mask_train, batch_size=2, epochs=3000, verbose=2, shuffle=True,
              validation_split=0.10, callbacks=[model_checkpoint, reduce_lr, model_es, csv_logger])

def predict(model):
    print('-'*30)
    print('Loading and preprocessing test data...')
    print('-'*30)

    imgs_test = np.load('./data_python/1CDT_Green_Red_FarRed_Annotated_FISH_Dilation4Conn1Iter_Testing_128by128_normalize.npy')
    imgs_mask_test = np.load('.//data_python/1CDT_Green_Red_FarRed_Annotated_FISH_Dilation4Conn1Iter_Testing_128by128_normalize_Mask.npy')
    imgs_test = imgs_test.astype('float32')

    #imgs_train = np.load('../data_python/1CDT_Green_Red_Annotated_FISH_Dilation8Conn1Iter_Training_128by128.npy')
    #imgs_train = imgs_train.astype('float32')
    #mean = np.mean(imgs_train)  # mean for data centering
    #std = np.std(imgs_train)  # std for data normalization
    #del imgs_train
    #imgs_test -= mean
    #imgs_test /= std

    print('-'*30)
    print('Loading saved weights...')
    print('-'*30)

    model = get_unet_short()
    model.load_weights(modelwtsfname)

    print('-'*30)
    print('Predicting masks on test data...')
    print('-'*30)
    imgs_test = np.expand_dims(imgs_test,3)
    imgs_mask_test = model.predict(imgs_test, batch_size=256,verbose=1)
    np.save('/data/gudlap/koh/spot_learners_deeplearning/data_python/1CDT_Green_Red_Annotated_FISH_NoDillations__Testing_128by128_normalize_Mask_Pred_short.npy', np.squeeze(imgs_mask_test))


def evaluate_params(args): 

    images , masks = get_images(args.images, args.masks, args.augmentation, args.normalize_mask)

    #Get the images size  
    img_rows = np.shape(images)[1]
    img_cols = np.shape(images)[2]
        
    #np.save("augmented-images.npy", np.squeeze(images))
    model = get_unet(img_rows, img_cols, n_layers=args.n_layers, \
                     dropout=args.dropout, \
                     filter_size=args.num_filters, \
                     activation_func=args.activation, \
                     conv_size=args.conv_size)

    history_callback = train(model, images, masks, initialize=args.initialize)
    return history_callback
 


if __name__ == '__main__':


    parser = argparse.ArgumentParser(description="spot learner")

    parser.add_argument('images',  help="The 2d numpy array image stack or 128 * 128")
    parser.add_argument('masks',  help="The 2d numpy array mask (8bits) stack or 128 * 128")
    parser.add_argument('--nlayers', default=5, type = int, dest='n_layers', help="The number of layer in the forward path ")
    parser.add_argument('--num_filters', default=32, type = int, dest='num_filters', help="The number of convolution filters in the first layer")
    parser.add_argument('--conv_size', default='3', type = int, dest='conv_size', help="The convolution filter size.")
    parser.add_argument('--dropout', default=None, type = float, dest='dropout', help="Include a droupout layer with a specific dropout value.")
    parser.add_argument('--activation', default='relu',dest='activation', help="Activation function.")
    parser.add_argument('--augmentation', default=None, type = float, dest='augmentation', help="Augmentation factor for the training set.")
    parser.add_argument('--initialize', default=None, dest='initialize', help="Numpy array for weights initialization.")
    parser.add_argument('--normalize_mask', action='store_true', dest='normalize_mask', help="Normalize the mask to 0-1 by dividing by 255.")

            
    args = parser.parse_args()
    history_callback = evaluate_params(args)
    print("Best val_dice_coef:")
    print(max(history_callback.history["val_dice_coef"]))
    #Save the history as pickle object
    pickle.dump(history_callback.history, open( "fit_history.p", "wb" ) )
#    predict()
