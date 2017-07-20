from __future__ import print_function

import os
import numpy as np
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, Dropout
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,EarlyStopping
from keras import backend as K
from keras.models import load_model
import argparse
import pickle


K.set_image_data_format('channels_last')  # TF dimension ordering in this code
__file__="/gpfs/gsfs2/users/zakigf/swift-t/examples/nci-hitif/model.py"
img_rows = 128
img_cols = 128

smooth = 1.

reloadFlag = True
savePNGS = False
#oldmodelwtsfname = 'short_weights_nodil_normalize.h5'
modelwtsfname = 'modle_weights.h5'


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)



def get_unet (n_layers=5, filter_size=32, dropout=True, activation_func="relu", conv_size=3 ):

    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    inputs = Input((img_rows, img_cols, 1))
    conv_layers=[] 
    pool_layers=[inputs]
    conv_filter=(conv_size, conv_size )


    for i in range(n_layers):
        conv = Conv2D(filter_size,  conv_filter, activation=activation_func, padding='same')(pool_layers[i])
        if dropout:
            conv = Dropout(0.2)(conv)
        conv = Conv2D(filter_size, conv_filter, activation=activation_func, padding='same')(conv)
        pool = MaxPooling2D(pool_size=(2, 2))(conv)
        conv_layers.append(conv)
        pool_layers.append(pool)
        filter_size *=2
        
  
    filter_size /=4

    for i in range(n_layers-1):
        up = concatenate([Conv2DTranspose(filter_size, (2, 2), strides=(2, 2), padding='same')(conv_layers[-1]), conv_layers[n_layers-i-2]], axis=3)



        conv = Conv2D(filter_size, conv_filter, activation=activation_func, padding='same')(up)
        if dropout:  
            conv = Dropout(0.2)(conv)
        conv = Conv2D(filter_size, conv_filter, activation=activation_func, padding='same')(conv)
        conv_layers.append(conv)
        filter_size /= 2


    last_conv =  Conv2D(1, (1, 1), activation='sigmoid')(conv_layers[-1])
    conv_layers.append(last_conv)

    model = Model(inputs=[inputs], outputs=[last_conv])


    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])
    model.summary()

    return model

def train(model):

    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)

    imgs_train_file="1CDT_Green_Red_FarRed_Annotated_FISH_Dilation4Conn1Iter_Training_128by128_normalize.npy"
    imgs_mask_file="1CDT_Green_Red_FarRed_Annotated_FISH_Dilation4Conn1Iter_Training_128by128_normalize_Mask.npy"
    file_path = os.path.dirname(os.path.realpath(__file__))
    imgs_train_path=os.path.join(file_path, "data_python", imgs_train_file)
    imgs_mask_path=os.path.join(file_path, "data_python", imgs_mask_file)


    imgs_train = np.load(imgs_train_path)
    imgs_mask_train = np.load(imgs_mask_path)

    #imgs_train = np.load('./data_python/1CDT_Green_Red_FarRed_Annotated_FISH_Dilation4Conn1Iter_Training_128by128_normalize.npy')
    #imgs_mask_train = np.load('./data_python/1CDT_Green_Red_FarRed_Annotated_FISH_Dilation4Conn1Iter_Training_128by128_normalize_Mask.npy')


    imgs_train = imgs_train.astype('float32')
    #mean = np.mean(imgs_train)  # mean for data centering
    #std = np.std(imgs_train)  # std for data normalization

    #imgs_train -= mean
    #imgs_train /= std

    imgs_mask_train = imgs_mask_train.astype('float32')
    imgs_mask_train /= 255.  # scale masks to [0, 1]

   
#    model = get_unet()
#
#    if os.path.isfile(oldmodelwtsfname) and reloadFlag :
#       print('-'*30)
#       print('Loading previous weights ...')
#       model.load_weights(oldmodelwtsfname)


# G.Z. already done when mode is passed 
#    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])
#    model.summary()

    model_checkpoint = ModelCheckpoint(modelwtsfname, monitor='val_loss', save_best_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,patience=50, min_lr=0.001,verbose=1)
    model_es = EarlyStopping(monitor='val_loss', min_delta=0.0000001, patience=100, verbose=1, mode='auto')

    print('-'*30)
    print('Fitting model...')
    print('-'*30)
    imgs_train = np.expand_dims(imgs_train, 3)
    imgs_mask_train = np.expand_dims(imgs_mask_train, 3)
    return model.fit(imgs_train, imgs_mask_train, batch_size=2, epochs=2, verbose=2, shuffle=True,
              validation_split=0.10, callbacks=[model_checkpoint, reduce_lr, model_es])

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

if __name__ == '__main__':


    parser = argparse.ArgumentParser(description="spot learner")


#def get_unet (n_layers=5, filter_size=32, dropout=True, activation_func="relu", conv_size=3 ):

    parser.add_argument('--nlayers', default=5, type = int, dest='n_layers', help="The number of layer in the forward path ")
    parser.add_argument('--num_filters', default=32, type = int, dest='num_filters', help="The number of convolution filters in the first layer")
    parser.add_argument('--conv_size', default='3', type = int, dest='conv_size', help="The convolution filter size.")
    parser.add_argument('--dropout', default=True, type = bool, dest='dropout', help="Include a droupout layer.")
    parser.add_argument('--activation', default='relu',dest='activation', help="Activation function.")

    args = parser.parse_args()

    model = get_unet(n_layers=args.n_layers, \
                     dropout=args.dropout, \
                     filter_size=args.num_filters, \
                     activation_func=args.activation, \
                     conv_size=args.conv_size)
    history_callback = train(model)
    print("Best val_dice_coef:")
    print(max(history_callback.history["val_dice_coef"]))
    #Save the history as pickle object
    pickle.dump(history_callback.history, open( "fit_history.p", "wb" ) )
#    predict()

