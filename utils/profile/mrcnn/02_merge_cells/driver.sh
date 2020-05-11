#!/bin/bash         
MRCNN_DEMO="../../../../framework-nucleus-segmentation/inference/mrcnn/demo/"
MRCNN_SRC="../../../../framework-nucleus-segmentation/inference/mrcnn/src/"
INF_SRC="../../../../framework-nucleus-segmentation/mrcnn/samples/cell/"
cp $MRCNN_DEMO/demo.py .
#cp $MRCNN_SRC/mrcnn_infer.py .
cp $INF_SRC/inference.py .
sed -i '/def stitched_inference(image, cropsize, model, padding=40)/i @profile' inference.py
sed -i 's/img = np.zeros((len(image_list),1078,1278))/img = np.zeros((100,1078,1278))/g' demo.py
sed -i 's/from mrcnn_infer import */from mrcnn_infer_profile import /g' demo.py
sed -i '/img_as_ubyte/d' demo.py
sed -i '/for i in range/i image_resized = img_as_ubyte(resize(np.array(Image.open(image_list[0])), (1078, 1278)))' demo.py
sed -i '/import sys,glob,warnings,os/i import line_profiler' demo.py
sed -i '/import line_profiler/i from skimage.util import img_as_ubyte' demo.py
#sed -i 's/from inference import/from inference_profile import/g' mrcnn_infer.py

mv demo.py demo_profile.py
#mv mrcnn_infer.py mrcnn_infer_profile.py
mv inference.py inference_profile.py

cp demo_profile.py $MRCNN_DEMO
cp mrcnn_infer_profile.py $MRCNN_SRC
cp inference_profile.py $INF_SRC

RES_FOLDER=`pwd`
pushd $MRCNN_DEMO
kernprof -l -v demo_profile.py > $RES_FOLDER/result.txt
popd
rm *.py
