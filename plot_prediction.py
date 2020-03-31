#!/usr/bin/python

from __future__ import print_function
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
# Kludge for raw_input Py2x to Py3x
from six.moves import input as raw_input

## Outline
#inputnpyfname ='/gpfs/gsfs10/users/HiTIF/progs/dl_segmentation/segmentation_models/Sigal_WellE03/input_augmentedintensity_rotations/20180823_075547/WellE03_chunkindex_010_DAPI_uint16touint8_normalizeandscale_dtypeuint8_patches.npy'
#targetnpyfname ='/gpfs/gsfs10/users/HiTIF/progs/dl_segmentation/segmentation_models/Sigal_WellE03/input_augmentedintensity_rotations/20180823_075547/WellE03_chunkindex_010_boundarydilate1_dtypebool_patches.npy'
#predictnpyfname = '/gpfs/gsfs10/users/HiTIF/progs/dl_segmentation/segmentation_models/Sigal_WellE03/input_augmentedintensity_rotations/20180823_075547/WellE03_chunkindex_010_boundarydilate1_dtypebool_patches_predict_UResNet152.npy'
#predictnpyfname='/gpfs/gsfs10/users/HiTIF/progs/dl_segmentation/segmentation_models/Sigal_WellE03/input_augmentedintensity_rotations/20180823_075547/WellE03_chunkindex_010_boundarydilate1_dtypebool_patches_predict_UResNet152_diceandbce.npy'
#predictnpyfname = '/gpfs/gsfs10/users/HiTIF/progs/dl_segmentation/segmentation_models/Sigal_WellE03/input_augmentedintensity_rotations/20180823_075547/WellE03_chunkindex_010_boundarydilate1_dtypebool_patches_predict_UResNet152_diceandbce_size1by1_size1by3.npy'
#predictnpyfname = '/gpfs/gsfs10/users/HiTIF/progs/dl_segmentation/segmentation_models/Sigal_WellE03/input_augmentedintensity_rotations/20180823_075547/WellE03_chunkindex_010_boundarydilate1_dtypebool_patches_predict_UResNet152FPN_diceandbce_size1by1_size1by3.npy'
#t_str = 'Outline'

## MaskUint8
#inputnpyfname ='/gpfs/gsfs10/users/HiTIF/progs/dl_segmentation/segmentation_models/Sigal_WellE03/input_augmentedintensity_rotations/20180823_075547/WellE03_chunkindex_011_DAPI_uint16touint8_normalizeandscale_dtypeuint8_patches.npy'
#targetnpyfname ='/gpfs/gsfs10/users/HiTIF/progs/dl_segmentation/segmentation_models/Sigal_WellE03/input_augmentedintensity_rotations/20180823_075547/WellE03_chunkindex_011_bitmask_dtypebool_patches.npy'
#predictnpyfname = '/gpfs/gsfs10/users/HiTIF/progs/dl_segmentation/segmentation_models/Sigal_WellE03/input_augmentedintensity_rotations/20180823_075547/WellE03_chunkindex_011_bitmask_dtypebool_patches_predict_UResNet152.npy'
#predictnpyfname = '/gpfs/gsfs10/users/HiTIF/progs/dl_segmentation/segmentation_models/Sigal_WellE03/input_augmentedintensity_rotations/20180823_075547/WellE03_chunkindex_011_bitmask_dtypebool_patches_predict_UResNet152FPN_size1by1_size1by3.npy'
#t_str = 'MaskUint8'

## MaskUint16
#inputnpyfname = '/gpfs/gsfs10/users/HiTIF/progs/dl_segmentation/segmentation_models/Sigal_WellE03/WellE03_batch2_Uint16DAPI_for_masks_float32_256by256_patches.npy'
#targetnpyfname = '/gpfs/gsfs10/users/HiTIF/progs/dl_segmentation/segmentation_models/Sigal_WellE03/WellE03_batch2_float32_Uint16DAPImasks_256by256_patches.npy'
#predictnpyfname = '/gpfs/gsfs10/users/HiTIF/progs/dl_segmentation/segmentation_models/Sigal_WellE03/output/WellE03_batch2_float32_Uint16DAPImasks_256by256_patches_predict_UResNet152.npy'
#t_str = 'MaskUint16'

## EDT
'''
inputnpyfname ='/gpfs/gsfs10/users/HiTIF/progs/dl_segmentation/segmentation_models/Sigal_WellE03/input_augmentedintensity_rotations/20180823_075547/WellE03_chunkindex_011_DAPI_uint16touint8_normalizeandscale_dtypeuint8_patches.npy'
targetnpyfname ='/gpfs/gsfs10/users/HiTIF/progs/dl_segmentation/segmentation_models/Sigal_WellE03/input_augmentedintensity_rotations/20180823_075547/WellE03_chunkindex_011_distancemapnormalized_dtypefloat32_patches.npy'
predictnpyfname = '/gpfs/gsfs10/users/HiTIF/progs/dl_segmentation/segmentation_models/Sigal_WellE03/input_augmentedintensity_rotations/20180823_075547/WellE03_chunkindex_011_distancemapnormalized_dtypefloat32_patches_predict_ResNet152FPN_size1by1_size1by3.npy'
t_str = 'EDT'
'''
## GaussianBlur
inputnpyfname ='/gpfs/gsfs10/users/HiTIF/progs/dl_segmentation/segmentation_models/Sigal_WellE03/input_augmentedintensity_rotations/20180823_075547/WellE03_chunkindex_011_DAPI_uint16touint8_normalizeandscale_dtypeuint8_patches.npy'
targetnpyfname ='/gpfs/gsfs10/users/HiTIF/progs/dl_segmentation/segmentation_models/Sigal_WellE03/input_intaugmentation_resize_patches/WellE03_MIP2DImages_CorrectedSegmentationMask_ForDL_UResnet152/gudlap/20181007_173118/WellE03_Iter0_ParaChunk_0_size1by1_float32_outlinegaussianblur_raidus1_images.npy'
predictnpyfname ='/gpfs/gsfs10/users/HiTIF/progs/dl_segmentation/segmentation_models/Sigal_WellE03/input_augmentedintensity_rotations/20180823_075547/WellE03_chunkindex_011_DAPI_uint16touint8_normalizeandscale_dtypeuint8_patches_IncResV2FPN_gblurradius1.npy'
t_str = 'GBlur-Radius1'

## GaussianBlur
#inputnpyfname ='/gpfs/gsfs10/users/HiTIF/progs/dl_segmentation/segmentation_models/Sigal_WellE03/input_intaugmentation_resize_patches/WellE03_MIP2DImages_CorrectedSegmentationMask_ForDL_UResnet152/gudlap/20181007_173118/WellE03_Iter0_ParaChunk_0_size1by1_256by256pacthes_normalizeandscaleuint16touint8_intensity_images.npy'
#targetnpyfname ='/gpfs/gsfs10/users/HiTIF/progs/dl_segmentation/segmentation_models/Sigal_WellE03/input_intaugmentation_resize_patches/WellE03_MIP2DImages_CorrectedSegmentationMask_ForDL_UResnet152/gudlap/20181007_173118/WellE03_Iter0_ParaChunk_0_size1by1_float32_outlinegaussianblur_radius1_normalize01_ROIs_images.npy'
#predictnpyfname = '/gpfs/gsfs10/users/HiTIF/progs/dl_segmentation/segmentation_models/Sigal_WellE03/input_intaugmentation_resize_patches/WellE03_MIP2DImages_CorrectedSegmentationMask_ForDL_UResnet152/gudlap/20181007_173118/WellE03_Iter0_ParaChunk_0_size1by1_float32_outlinegaussianblur_radius1_normalize01_ROIs_images.npy'
#t_str = 'GBlur-Radius1'

## GaussianBlur-OutsideNucleus
#inputnpyfname = '/gpfs/gsfs10/users/HiTIF/progs/dl_segmentation/segmentation_models/Sigal_WellE03/WellE03_batch2_DAPI_for_gaussianblur_radius3_outsidenucleus_float32_256by256_patches.npy'
#targetnpyfname = '/gpfs/gsfs10/users/HiTIF/progs/dl_segmentation/segmentation_models/Sigal_WellE03/WellE03_batch2_float32_gaussianblur_radius3_outsidenucleus_256by256_patches.npy'
#predictnpyfname = '/gpfs/gsfs10/users/HiTIF/progs/dl_segmentation/segmentation_models/Sigal_WellE03/output/WellE03_batch2_float32_gaussianblur_radius3_outsidenucleus_256by256_patches_predict_UResNet152.npy'
#t_str = 'GBlur-Radius3-Outside'

## GaussianBlur-OnandInsideNucleus
#inputnpyfname = '/gpfs/gsfs10/users/HiTIF/progs/dl_segmentation/segmentation_models/Sigal_WellE03/WellE03_batch2_DAPI_for_gaussianblur_radius3_insidenucleus_float32_256by256_patches.npy'
#targetnpyfname = '/gpfs/gsfs10/users/HiTIF/progs/dl_segmentation/segmentation_models/Sigal_WellE03/WellE03_batch2_float32_gaussianblur_radius3_insidenucleus_256by256_patches.npy'
#predictnpyfname = '/gpfs/gsfs10/users/HiTIF/progs/dl_segmentation/segmentation_models/Sigal_WellE03/output/WellE03_batch2_float32_gaussianblur_radius3_insidenucleus_256by256_patches_predict_UResNet152.npy'
#t_str = 'GBlur-Radius3-OnandInside'

plotFlag = True

if plotFlag : 
   
  imgs_train = np.load(inputnpyfname).astype('float32')
  imgs_mask_train = np.load(targetnpyfname).astype('float32')
  imgs_mask_predict = np.load(predictnpyfname).astype('float32')
  print('Shape of input, target and prediction images ', imgs_train.shape, imgs_mask_train.shape, imgs_mask_predict.shape)    
  print('Input images min, max before normalization ', np.amin(imgs_train),np.amax(imgs_train))
  print('Target images min, max before normalization ', np.min(imgs_mask_train),np.max(imgs_mask_train))
  print('Prediction images min, max before normalization ', np.min(imgs_mask_predict),np.max(imgs_mask_predict))
  imgs_train = imgs_train/255.0  
  print('Input images min, max after normalization ', np.amin(imgs_train),np.amax(imgs_train))
  plt.ion()
  for i in range(0,imgs_train.shape[0]):
  #for i in range(imgs_train.shape[0]-1,-1,-1):
      img = np.squeeze(imgs_train[i,:,:])
      img_amax = np.amax(img)
      img_amin = np.amin(img)
	  
      mask = np.squeeze(imgs_mask_train[i,:,:])
      mask_amax = np.amax(mask)
      mask_amin = np.amin(mask)
      mask_pre = np.squeeze(imgs_mask_predict[i,:,:])
      mask_pre_amax = np.amax(mask_pre)
      mask_pre_amin = np.amin(mask_pre)
	  
      fig = plt.figure(1)
      a = fig.add_subplot(1,4,1)
      imgplot = plt.imshow(img)
      imgplot.set_clim(img_amin,img_amax)
      a.set_title('Input : '+str(i))
      plt.colorbar(orientation ='horizontal')
      a = fig.add_subplot(1,4,2)
      imgplot = plt.imshow(mask)
      imgplot.set_clim(mask_amin, mask_amax)
      a.set_title(t_str + ' ' +str(i))
      plt.colorbar(orientation ='horizontal')
      a = fig.add_subplot(1,4,3)
      imgplot = plt.imshow(mask_pre)
      imgplot.set_clim(mask_pre_amin, mask_pre_amax)
      a.set_title(t_str + ' Prediction : '+str(i))
      plt.colorbar(orientation ='horizontal')

      a = fig.add_subplot(1,4,4)
      diff_img = mask_pre-mask
      imgplot = plt.imshow(diff_img)
      diff_amin = np.amin(diff_img)
      diff_amax = np.amax(diff_img)	  
      imgplot.set_clim(diff_amin, diff_amax)
      a.set_title('Prediction-Input '+ t_str +' : '+str(i))
      plt.colorbar(orientation ='horizontal')

      plt.show()
      _ = raw_input("Press [enter] to continue.")      
      #plt.pause(2)
      plt.draw()
      plt.clf()

