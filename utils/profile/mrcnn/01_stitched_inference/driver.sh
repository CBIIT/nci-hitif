#!/bin/bash          
INF_SRC="../../../../framework-nucleus-segmentation/mrcnn/samples/cell/inference.py"
INF_PROFILE="../../../../framework-nucleus-segmentation/mrcnn/samples/cell/inference_profile.py"

rm -f $INF_PROFILE
cp $INF_SRC $INF_PROFILE
sed  '/\[option\]/i Hello World' input a.txt
