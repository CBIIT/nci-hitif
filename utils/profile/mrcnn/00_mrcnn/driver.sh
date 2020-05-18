#!/bin/bash

# Activate Conda Environment
source /data/HiTIF/progs/miniconda/miniconda3_setup_script.sh
conda activate hitif

# Output file location configuration
MRCNN_DEMO="../../../../framework-nucleus-segmentation/inference/mrcnn/demo/"
MRCNN_SRC="../../../../framework-nucleus-segmentation/inference/mrcnn/src/"
INF_SRC="../../../../framework-nucleus-segmentation/mrcnn/samples/cell/"

# Run utility for getting new python codes up to the configuration file.
python3 ../util/profile_util.py config.ini

cp demo_profile.py $MRCNN_DEMO
cp mrcnn_infer_profile.py $MRCNN_SRC
cp inference_profile.py $INF_SRC

# Execution and Store Benchmark Result.
RES_FOLDER=`pwd`
pushd $MRCNN_DEMO
python3 demo_profile.py > $RES_FOLDER/result.txt
popd

# Clean up generated python codes.
rm *.py
rm $MRCNN_DEMO/demo_profile.py
rm $MRCNN_SRC/mrcnn_infer_profile.py
rm $INF_SRC/inference_profile.py
