#!/bin/bash         
MRCNN_DEMO="/gpfs/gsfs10/users/HiTIF/data/kyunghhun/nci-hitif/framework-nucleus-segmentation/inference/mrcnn/demo/"
MRCNN_SRC="/gpfs/gsfs10/users/HiTIF/data/kyunghhun/nci-hitif/framework-nucleus-segmentation/inference/mrcnn/src/"
cp $MRCNN_DEMO/demo.py .
cp $MRCNN_SRC/mrcnn_infer.py .
sed -i '/def mrcnn_infer(img, mrcnn_model_path, config_file_path):/i @profile' mrcnn_infer.py
sed -i 's/img = np.zeros((len(image_list),1078,1278))/img = np.zeros((1,1078,1278))/g' demo.py
sed -i 's/from mrcnn_infer import */from mrcnn_infer_profile import /g' demo.py
sed -i '/img_as_ubyte/d' demo.py
sed -i '/for i in range/i image_resized = img_as_ubyte(resize(np.array(Image.open(image_list[0])), (1078, 1278)))' demo.py
sed -i '/import sys,glob,warnings,os/i import line_profiler' demo.py
sed -i '/import line_profiler/i from skimage.util import img_as_ubyte' demo.py
mv demo.py demo_profile.py
mv mrcnn_infer.py mrcnn_infer_profile.py

cp demo_profile.py $MRCNN_DEMO
cp mrcnn_infer_profile.py $MRCNN_SRC

RES_FOLDER=`pwd`
pushd $MRCNN_DEMO
kernprof -l -v demo_profile.py > $RES_FOLDER/result.txt
popd
