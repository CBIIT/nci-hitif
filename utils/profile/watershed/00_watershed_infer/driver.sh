#!/bin/bash

# Activate Conda Environment
source /data/HiTIF/progs/miniconda/miniconda3_setup_script.sh
conda activate hitif

# Output file location configuration
WATERSHED_DEMO="../../../../framework-nucleus-segmentation/inference/watershed/demo/"
WATERSHED_SRC="../../../../framework-nucleus-segmentation/inference/watershed/src/"

# Copy input files
cp ../util/input/*.py .

# Run utility for getting new python codes up to the configuration file.
python3 ../util/profile_util.py config.ini

cp demo_profile.py $WATERSHED_DEMO
cp watershed_infer_profile.py $WATERSHED_SRC

# Execution and Store Benchmark Result.
RES_FOLDER=`pwd`
pushd $WATERSHED_DEMO
python3 demo_profile.py > $RES_FOLDER/result.txt
popd

# remove redundant part of result.txt
sed -i '/Line #      Hits         Time  Per Hit   % Time  Line Contents/,$!d' result.txt

# Clean up generated python codes.
rm *.py
rm $WATERSHED_DEMO/demo_profile.py
rm $WATERSHED_SRC/watershed_infer_profile.py
