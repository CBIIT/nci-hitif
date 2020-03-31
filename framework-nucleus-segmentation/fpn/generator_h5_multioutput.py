from __future__ import print_function
import numpy as np
import h5py
import keras
from keras import backend as K
K.set_image_data_format('channels_last')  # TF dimension ordering in this code
import random

class DataGeneratorH5(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, source_target_list_IDs, h5fname, source_datasetname, target_datasetname, batch_size=16, dim=(256,256), n_in_channels=3,n_out_channels=1,shuffle=True, scale_source=True, scale_target=True, target_abs_value=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.source_target_list_IDs = source_target_list_IDs
        self.h5fname = h5fname
        self.source_datasetname = source_datasetname
        self.target_datasetname = target_datasetname
        self.n_in_channels = n_in_channels
        self.n_out_channels = n_out_channels
        self.shuffle = shuffle
        self.scale_source = scale_source
        self.scale_target = scale_target
        self.target_abs_value = target_abs_value
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor( (len(self.source_target_list_IDs) / self.batch_size)))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        #indexes = self.indexes[index:(index+1)]

        # Find list of source and IDs
        src_tar_list_IDs_temp = [self.source_target_list_IDs[k] for k in indexes]
        np.sort(src_tar_list_IDs_temp)
        #print(len(src_tar_list_IDs_temp))

        # Generate data
        X, Y = self.__data_generation(src_tar_list_IDs_temp)

        return X, Y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.source_target_list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, src_tar_list_IDs_temp):
        sorted_list = src_tar_list_IDs_temp
        np.sort(sorted_list)
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_in_channels), Y: (n_samples, *dim, n_out_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_in_channels), dtype='float32')
        Y = np.empty((self.batch_size, *self.dim, self.n_out_channels), dtype='float32')
        
        with h5py.File(self.h5fname, 'r') as hf:
             i = 0
             for j in sorted_list:
                X[i,:,:,0] = hf[self.source_datasetname[0]][j].astype('float32')
                X[i,:,:,1] = hf[self.source_datasetname[0]][j].astype('float32')
                X[i,:,:,2] = hf[self.source_datasetname[0]][j].astype('float32')      
                for k in range(len(self.target_datasetname)):				
                    Y[i,:,:,k] = hf[self.target_datasetname[k]][j]
                i = i+1
        if self.scale_source :
            X =  X/255.0
        if self.scale_target :
            Y =  Y.astype('float32')
        if self.target_abs_value :
            Y = np.abs(Y)
        #print(':: Size of Source Batch Images = ', X.shape, " min = ", np.min(X), " max = ", np.max(X))
        #print(':: Size of Target Batch Images = ', Y.shape, " min = ", np.min(Y), " max = ", np.max(Y))
        return X, Y