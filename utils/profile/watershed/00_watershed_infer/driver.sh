#!/bin/bash         
WATERSHED_DEMO="/gpfs/gsfs10/users/HiTIF/data/kyunghhun/nci-hitif/framework-nucleus-segmentation/inference/watershed/demo/"
WATERSHED_SRC="/gpfs/gsfs10/users/HiTIF/data/kyunghhun/nci-hitif/framework-nucleus-segmentation/inference/watershed/src/"
cp $WATERSHED_DEMO/demo.py .
cp $WATERSHED_SRC/watershed_infer.py .
sed -i '/def watershed_infer(img,gaussian_blur_model,distance_map_model,config_file_path):/i @profile' watershed_infer.py
sed -i 's/img = np.zeros((len(image_list),1078,1278))/img = np.zeros((1,1078,1278))/g' demo.py
sed -i 's/from watershed_infer import */from watershed_infer_profile import /g' demo.py
sed -i '/import sys,glob,warnings,os/i import line_profiler' demo.py
sed -i '/image_resized = img_as_ubyte/d' demo.py
sed -i '/img = np.zeros((1,1078,1278))/i image_resized = img_as_ubyte(resize(np.array(Image.open(image_list[0])), (1078, 1278)))' demo.py
mv demo.py demo_profile.py
mv watershed_infer.py watershed_infer_profile.py

cp demo_profile.py $WATERSHED_DEMO
cp watershed_infer_profile.py $WATERSHED_SRC

RES_FOLDER=`pwd`
pushd $WATERSHED_DEMO
kernprof -l -v demo_profile.py > $RES_FOLDER/result.txt
popd
