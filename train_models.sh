#!/usr/bin/bash
module load CUDA/9.2 cuDNN/7.5/CUDA-9.2; \
source /data/HiTIF/progs/miniconda/miniconda2/bin/activate segmentation_model; \
cd /gpfs/gsfs10/users/HiTIF/data/dl_segmentation_paper/code/python/unet_fpn; \
#
##################### Absolute-GaussianBlur-Radius1-InceptionResNetV2-FPN  with Sigmoid activation and MSE ##############################
#
python -u train_test_unets_fpns.py \
--gpuPCIID 0 \
--expandInputChannels   \
--numberofTargetChannels 1   \
--freezeEncoderBackbone   \
--architectureType "FPN"   \
--backboneEncoderWeightsName "imagenet"   \
--backboneName "inceptionresnetv2"   \
--activationFunction "sigmoid"   \
--outputModelPrefix "IncResV2FPN"   \
--outputModelSuffix "input_normalizeandscaleUint8_target_abs_gaussianblur_radius1_loss_mse"   \
--trainingBatchSize 16   \
--inferenceBatchSize 8   \
--loss_function "mean_squared_error"   \
--numCSEpochs 2   \
--numFTEpochs 25   \
--h5fname "/gpfs/gsfs10/users/HiTIF/data/dl_segmentation_paper/data/biorep1_mcf10a_welle03/augmented_patches/mcf10a_biorep1_welle03_60x_bin2/gudlap/20190523_124204/mcf10a_biorep1_welle03_babe_60x2_Resize_Factors_1p00_2p00_0p33_0p67_1p33_input_gt_derived_outputs.h5"   \
--srcDatasetName "DAPI_uint16touint8_normalizeandscale"   \
--tarDatasetName "gblurradius1_float32"   \
--useTTAForTesting   \
--testinputnumpyfname '/gpfs/gsfs10/users/HiTIF/progs/dl_segmentation/segmentation_models/Sigal_WellE03/input_augmentedintensity_rotations/20180823_075547/WellE03_chunkindex_011_DAPI_uint16touint8_normalizeandscale_dtypeuint8_patches.npy'   \
--testpredictnumpyfname '/gpfs/gsfs10/users/HiTIF/progs/dl_segmentation/segmentation_models/Sigal_WellE03/input_augmentedintensity_rotations/20180823_075547/WellE03_chunkindex_011_DAPI_uint16touint8_normalizeandscale_dtypeuint8_patches_IncResV2FPN_gblurradius1.npy' >& ./logs/training_gblur_radius1_incresv2_fpn.log & \
#
##################### normalizedEDT-InceptionResNetV2-FPN  with Sigmoid activation and MSE ##############################
#
python -u train_test_unets_fpns.py \
--gpuPCIID 1 \
--expandInputChannels   \
--numberofTargetChannels 1   \
--freezeEncoderBackbone   \
--architectureType "FPN"   \
--backboneEncoderWeightsName "imagenet"   \
--backboneName "inceptionresnetv2"   \
--activationFunction "sigmoid"   \
--outputModelPrefix "IncResV2FPN"   \
--outputModelSuffix "input_normalizeandscaleUint8_target_nedt_loss_mse"   \
--trainingBatchSize 16   \
--inferenceBatchSize 8   \
--loss_function "mean_squared_error"   \
--numCSEpochs 2   \
--numFTEpochs 25   \
--h5fname "/gpfs/gsfs10/users/HiTIF/data/dl_segmentation_paper/data/biorep1_mcf10a_welle03/augmented_patches/mcf10a_biorep1_welle03_60x_bin2/gudlap/20190523_124204/mcf10a_biorep1_welle03_babe_60x2_Resize_Factors_1p00_2p00_0p33_0p67_1p33_input_gt_derived_outputs.h5"   \
--srcDatasetName "DAPI_uint16touint8_normalizeandscale"   \
--tarDatasetName "distancemapnormalized_float32"   \
--useTTAForTesting   \
--testinputnumpyfname '/gpfs/gsfs10/users/HiTIF/progs/dl_segmentation/segmentation_models/Sigal_WellE03/input_augmentedintensity_rotations/20180823_075547/WellE03_chunkindex_011_DAPI_uint16touint8_normalizeandscale_dtypeuint8_patches.npy'   \
--testpredictnumpyfname '/gpfs/gsfs10/users/HiTIF/progs/dl_segmentation/segmentation_models/Sigal_WellE03/input_augmentedintensity_rotations/20180823_075547/WellE03_chunkindex_011_DAPI_uint16touint8_normalizeandscale_dtypeuint8_patches_IncResV2FPN_nedt.npy' >& ./logs/training_nedt_incresv2_fpn.log & \
#
##################### Outline-InceptionResNetV2-FPN  with Sigmoid activation and dice_coef_loss_bce ##############################
#
python -u train_test_unets_fpns.py \
--gpuPCIID 2 \
--expandInputChannels   \
--numberofTargetChannels 1   \
--freezeEncoderBackbone   \
--architectureType "FPN"   \
--backboneEncoderWeightsName "imagenet"   \
--backboneName "inceptionresnetv2"   \
--activationFunction "sigmoid"   \
--outputModelPrefix "IncResV2FPN"   \
--outputModelSuffix "input_normalizeandscaleUint8_target_outline_loss_dicebce"   \
--trainingBatchSize 16   \
--inferenceBatchSize 8   \
--loss_function "dice_coef_loss_bce"   \
--numCSEpochs 2   \
--numFTEpochs 25   \
--h5fname "/gpfs/gsfs10/users/HiTIF/data/dl_segmentation_paper/data/biorep1_mcf10a_welle03/augmented_patches/mcf10a_biorep1_welle03_60x_bin2/gudlap/20190523_124204/mcf10a_biorep1_welle03_babe_60x2_Resize_Factors_1p00_2p00_0p33_0p67_1p33_input_gt_derived_outputs.h5"   \
--srcDatasetName "DAPI_uint16touint8_normalizeandscale"   \
--tarDatasetName "alongboundary_bool"   \
--useTTAForTesting   \
--testinputnumpyfname '/gpfs/gsfs10/users/HiTIF/progs/dl_segmentation/segmentation_models/Sigal_WellE03/input_augmentedintensity_rotations/20180823_075547/WellE03_chunkindex_011_DAPI_uint16touint8_normalizeandscale_dtypeuint8_patches.npy'   \
--testpredictnumpyfname '/gpfs/gsfs10/users/HiTIF/progs/dl_segmentation/segmentation_models/Sigal_WellE03/input_augmentedintensity_rotations/20180823_075547/WellE03_chunkindex_011_DAPI_uint16touint8_normalizeandscale_dtypeuint8_patches_IncResV2FPN_outline.npy' >& ./logs/training_outline_incresv2_fpn.log & \
#
##################### Mask-InceptionResNetV2-FPN  with Sigmoid activation and dice_coef_loss_bce ##############################
#
python -u train_test_unets_fpns.py \
--gpuPCIID 3 \
--expandInputChannels   \
--numberofTargetChannels 1   \
--freezeEncoderBackbone   \
--architectureType "FPN"   \
--backboneEncoderWeightsName "imagenet"   \
--backboneName "inceptionresnetv2"   \
--activationFunction "sigmoid"   \
--outputModelPrefix "IncResV2FPN"   \
--outputModelSuffix "input_normalizeandscaleUint8_target_bitmask_loss_dicebce"   \
--trainingBatchSize 16   \
--inferenceBatchSize 8   \
--loss_function "dice_coef_loss_bce"   \
--h5fname "/gpfs/gsfs10/users/HiTIF/data/dl_segmentation_paper/data/biorep1_mcf10a_welle03/augmented_patches/mcf10a_biorep1_welle03_60x_bin2/gudlap/20190523_124204/mcf10a_biorep1_welle03_babe_60x2_Resize_Factors_1p00_2p00_0p33_0p67_1p33_input_gt_derived_outputs.h5"   \
--numCSEpochs 2   \
--numFTEpochs 25   \
--srcDatasetName "DAPI_uint16touint8_normalizeandscale"   \
--tarDatasetName "bitmask_bool"   \
--useTTAForTesting   \
--testinputnumpyfname '/gpfs/gsfs10/users/HiTIF/progs/dl_segmentation/segmentation_models/Sigal_WellE03/input_augmentedintensity_rotations/20180823_075547/WellE03_chunkindex_011_DAPI_uint16touint8_normalizeandscale_dtypeuint8_patches.npy'   \
--testpredictnumpyfname '/gpfs/gsfs10/users/HiTIF/progs/dl_segmentation/segmentation_models/Sigal_WellE03/input_augmentedintensity_rotations/20180823_075547/WellE03_chunkindex_011_DAPI_uint16touint8_normalizeandscale_dtypeuint8_patches_IncResV2FPN_bitmask.npy' >& ./logs/training_bitmask_incresv2_fpn.log & \
#
# deactivate
#
source /data/HiTIF/progs/miniconda/miniconda2/bin/deactivate segmentation_model;