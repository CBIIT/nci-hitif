#!/bin/bash

#module load CUDA
#module load cuDNN/7.0/CUDA-9.0
pushd /data/zakigf/conda/
source etc/profile.d/conda.sh 
popd
pushd /data/zakigf/mask-rcnn/
conda activate mask_rcnn
popd

#source /data/$whoami/conda/etc/profile.d/conda.sh
#conda activate mask_rcnn

#h5_path='/gpfs/gsfs10/users/HiTIF/progs/dl_segmentation/segmentation_models/Sigal_WellE03/input_intaugmentation_resize_patches/WellE03_MIP2DImages_CorrectedSegmentationMask_ForDL_UResnet152/gudlap/20181007_173118_merged/WellE03_DAPI_uint16touint8_normalizeandscale_dtypeuint8_patches.h5'
h5_path='/gpfs/gsfs10/users/HiTIF/progs/dl_segmentation/segmentation_models/Sigal_WellE03/input_intaugmentation_resize_patches/WellE03_MIP2DImages_CorrectedSegmentationMask_ForDL_UResnet152/gudlap/20181007_173118_merged/WellE03_DAPI_uint16touint8_normalizeandscale_dtypeuint8_patches_MASKRCNN.h5'


python cell_script.py train --dataset=$h5_path --init=coco --logs=/data/zakigf/mask-rcnn/Mask_RCNN/images/cell-images/training_weights

echo 'Done'
