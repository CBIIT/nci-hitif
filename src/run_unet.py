from __future__ import print_function

import os
import numpy as np
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,EarlyStopping, CSVLogger, Callback
from keras import backend as K
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
import argparse
import pickle
from utils import augmented_predict
import tensorflow as tf
import arguments




modelwtsfname = 'model_weights.h5'

class TestCallback(Callback):
    def __init__(self, test_data):
        self.test_data = test_data

    def on_epoch_end(self, epoch, logs={}):

        x, y = self.test_data
        loss, acc = self.model.evaluate(x, y, verbose=0, batch_size = 1)
        print('\nTesting loss: {}, acc: {}\n'.format(loss, acc))


def focal_loss(labels, logits, gamma=0, alpha=1.0):
    """
    focal loss for multi-classification
    FL(p_t)=-alpha(1-p_t)^{gamma}ln(p_t)
    Notice: logits is probability after softmax
    gradient is d(Fl)/d(p_t) not d(Fl)/d(x) as described in paper
    d(Fl)/d(p_t) * [p_t(1-p_t)] = d(Fl)/d(x)

    Focal Loss for Dense Object Detection, 
    https://doi.org/10.1016/j.ajodo.2005.02.022

    :param labels: ground truth labels, shape of [batch_size]
    :param logits: model's output, shape of [batch_size, num_cls]
    :param gamma:
    :param alpha:
    :return: shape of [batch_size]
    """
    epsilon = 1.e-9
    labels = tf.to_int64(labels)
    labels = tf.convert_to_tensor(labels, tf.int64)
    logits = tf.convert_to_tensor(logits, tf.float32)
    num_cls = logits.shape[1]

    model_out = tf.add(logits, epsilon)
    onehot_labels = tf.one_hot(labels, num_cls)
    ce = tf.multiply(onehot_labels, -tf.log(model_out))
    weight = tf.multiply(onehot_labels, tf.pow(tf.subtract(1., model_out), gamma))
    fl = tf.multiply(alpha, tf.multiply(weight, ce))
    reduced_fl = tf.reduce_max(fl, axis=1)
    # reduced_fl = tf.reduce_sum(fl, axis=1)  # same as reduce_max
    return reduced_fl


def dice_coef(y_true, y_pred):

    smooth = 1.
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3]) 
    dc = K.mean( (2. * intersection + smooth) / (union + smooth), axis=0)
    return dc

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def get_unet (img_rows, img_cols, args):

    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    print (img_rows)
    print (img_cols)

    print (args)
    n_layers = args.nlayers
    filter_size = args.num_filters
    dropout = args.dropout
    activation_func = args.activation
    conv_size = args.conv_size
    loss_func = args.loss_func
    last_activation = args.last_act
    batch_norm = args.batch_norm
    learning_rate = args.lr


    inputs = Input((img_rows, img_cols, 1))
    conv_layers=[] 
    pool_layers=[inputs]
    conv_filter=(conv_size, conv_size )


    for i in range(n_layers):
        conv = Conv2D(filter_size,  conv_filter, activation=activation_func, padding='same')(pool_layers[i])
        conv = BatchNormalization()(conv) if batch_norm else conv
        if dropout != None:
            conv = Dropout(dropout)(conv)
        conv = Conv2D(filter_size, conv_filter, activation=activation_func, padding='same')(conv)
        conv = BatchNormalization()(conv) if batch_norm else conv
        pool = MaxPooling2D(pool_size=(2, 2))(conv)
        conv_layers.append(conv)
        pool_layers.append(pool)
        filter_size *=2
        
  
    filter_size /=4

    for i in range(n_layers-1):
        up = concatenate([Conv2DTranspose(filter_size, (2, 2), strides=(2, 2), padding='same')(conv_layers[-1]), conv_layers[n_layers-i-2]], axis=3)
        conv = Conv2D(filter_size, conv_filter, activation=activation_func, padding='same')(up)
        conv = BatchNormalization()(conv) if batch_norm else conv
        if dropout != None:  
            conv = Dropout(dropout)(conv)
        conv = Conv2D(filter_size, conv_filter, activation=activation_func, padding='same')(conv)
        conv = BatchNormalization()(conv) if batch_norm else conv
        conv_layers.append(conv)
        filter_size /= 2


    #For binary classification, last activation should be sigmoid. 
#    if loss_func  == 'dice':
#        last_activation = 'sigmoid'
#    else:
#        print ("WARNING: last_activation set to None")
#        last_activation = None

    last_conv =  Conv2D(1, (1, 1), activation=last_activation)(conv_layers[-1])
    conv_layers.append(last_conv)

    model = Model(inputs=[inputs], outputs=[last_conv])

    if loss_func == 'dice':
        model.compile(optimizer=Adam(lr=learning_rate), loss=dice_coef_loss, metrics=[dice_coef])
    else:
        #Any Keras loss function will be passed
        model.compile(optimizer=Adam(lr=learning_rate), loss = loss_func)
    model.summary()
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    return model

def preprocess_images(images):
    imgs_train = np.squeeze(np.load(images))
    if imgs_train.ndim != 3:
        raise Exception("Error: The number of dimensions for images should equal 3, after squeezing the shape is:{0}".format(np.shape(images)))
    imgs_train = imgs_train.astype('float32')
    print("MAX before:{0}".format(np.amax(imgs_train)))
    #Normalize all number between 0 and 1.
    uint16_info = np.iinfo('uint16')
    imgs_train = imgs_train / uint16_info.max
    print("MAX after:{0}".format(np.amax(imgs_train)))
    imgs_train = np.expand_dims(imgs_train, axis= 3)

    return imgs_train


def preprocess_masks(masks, normalize_mask=False):
    
    imgs_mask_train = np.squeeze(np.load(masks))
    if imgs_mask_train.ndim != 3:
        raise Exception("Error: The number of dimensions for masks should equal 3, after squeezing the shape is:{0}".format(np.shape(masks)))

    imgs_mask_train = imgs_mask_train.astype('float32')
    if normalize_mask:
        imgs_mask_train /= 255.  # scale masks to [0, 1]

    imgs_mask_train = np.expand_dims(imgs_mask_train, axis= 3)
    return imgs_mask_train


def get_images(images, masks, augmentation_factor = 1, normalize_mask=False):

    print('-'*30)
    print('Loading and preprocessing train data...')
    print('-'*30)

#    imgs_train = np.squeeze(np.load(images))
#    imgs_mask_train = np.squeeze(np.load(masks))
#
#
#    if imgs_train.ndim != 3:
#        raise Exception("Error: The number of dimensions for images should equal 3, after squeezing the shape is:{0}".format(np.shape(images)))
#
#    if imgs_mask_train.ndim != 3:
#        raise Exception("Error: The number of dimensions for masks should equal 3, after squeezing the shape is:{0}".format(np.shape(masks)))
#
#    imgs_train = imgs_train.astype('float32')
#    imgs_mask_train = imgs_mask_train.astype('float32')
#    if normalize_mask:
#        imgs_mask_train /= 255.  # scale masks to [0, 1]
#
#
#    imgs_train = np.expand_dims(imgs_train, axis= 3)
#    imgs_mask_train = np.expand_dims(imgs_mask_train, axis= 3)

    imgs_train = preprocess_images(images) 
    imgs_mask_train = preprocess_masks(masks, normalize_mask) 

    #Shuffle the images
    np.random.seed(10)
    shuffled_id = np.random.permutation(imgs_train.shape[0])
    imgs_train = imgs_train[shuffled_id]
    imgs_mask_train = imgs_mask_train[shuffled_id]

    assert(np.amax(imgs_mask_train) <= 1)
    assert(np.amin(imgs_mask_train) >=  0)
    return_images = imgs_train 
    return_masks = imgs_mask_train 

    if augmentation_factor > 1:
    
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
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,patience=100, verbose=1)
    model_es = EarlyStopping(monitor='val_loss', min_delta=0.000001, patience=400, verbose=1, mode='auto')
    csv_logger = CSVLogger('training.csv')

    print('-'*30)
    print('Fitting model...')
    print('-'*30)

    if initialize != None:
        print("Initializing the model using:{0}\n", initialize)
        model.load_weights(initialize)

    
    test_call=TestCallback((imgs_train,imgs_mask_train))

    print (np.shape(imgs_train))
    print (np.shape(imgs_mask_train))
    #return model.fit(imgs_train, imgs_mask_train, batch_size=2, epochs=3000, verbose=2, shuffle=True,
    return model.fit(imgs_train, imgs_mask_train, batch_size=2, epochs=1500, verbose=2, shuffle=True,
              validation_split=0.10, callbacks=[model_checkpoint, reduce_lr, model_es, csv_logger])

def predict(model, weights, images, args):
    print('-'*30)
    print('Loading and preprocessing test data...')
    print('-'*30)

    #imgs_test = np.load('./data_python/1CDT_Green_Red_FarRed_Annotated_FISH_Dilation4Conn1Iter_Testing_128by128_normalize.npy')
    #imgs_mask_test = np.load('.//data_python/1CDT_Green_Red_FarRed_Annotated_FISH_Dilation4Conn1Iter_Testing_128by128_normalize_Mask.npy')
    #imgs_test = imgs_test.astype('float32')

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

    model.load_weights(weights)

    print('-'*30)
    print('Predicting masks on test data...')
    print('-'*30)
    #imgs_test = np.expand_dims(imgs_test,3)

    augment_prediction = args.rotate

    #augment the images if necessary
    if augment_prediction:
        aug_pred = augmented_predict(images) 
        aug_pred.augment_inputs()
        aug_images = aug_pred.get_augmented_patch()
        images = aug_images       
    print ('{0}'.format(np.shape(images)))
    print ('{0}'.format(type(images)))


    print("Inference images histogram") 
    hist, bin_edges = np.histogram(images)
    print(hist)
    print(bin_edges)

    imgs_mask_test = model.predict(images, batch_size = 1,verbose=1)

    #reduce the prediction if they were augmented
    if augment_prediction:
        imgs_mask_test = aug_pred.reduce_predictions(imgs_mask_test)

    print("Inference predictions histogram") 
    hist, bin_edges = np.histogram(imgs_mask_test)
    print(hist)
    print(bin_edges)
    
    np.save('mask_predictions.npy', np.squeeze(imgs_mask_test))

def evaluate_params(args): 

    images , masks = get_images(args.images, args.masks, args.augmentation, args.normalize_mask)

    print("Training images histogram") 
    hist, bin_edges = np.histogram(images)
    print(hist)
    print(bin_edges)

    print("Training masks histogram") 
    hist, bin_edges = np.histogram(masks)
    print(hist)
    print(bin_edges)


    #Get the images size  
    img_rows = np.shape(images)[1]
    img_cols = np.shape(images)[2]
        
    #np.save("augmented-images.npy", np.squeeze(images))
    model = get_unet(img_rows, img_cols, args)

    history_callback = train(model, images, masks, initialize=args.initialize)
    return history_callback
 


if __name__ == '__main__':

    parser = arguments.get_unet_parser()
            
    args = parser.parse_args()
    if not args.predict:
        history_callback = evaluate_params(args)
        print("Minimum validation loss:")
        print(min(history_callback.history["val_loss"]))
        #Save the history as pickle object
        pickle.dump(history_callback.history, open( "fit_history.p", "wb" ) )
    else:
        if args.initialize is None:
            raise Exception("Error: The initialize option has to be set for prediction")

        #It is not necessary to pass masks for prediction, but I am just following the function
        #prototype for now.
        images = preprocess_images(args.images)
        #Get the images size  
        img_rows = np.shape(images)[1]
        img_cols = np.shape(images)[2]
        model = get_unet(img_rows, img_cols, args)
        weights = args.initialize
        predict(model, weights, images, args)
