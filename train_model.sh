#!/usr/bin/bash
module load CUDA/9.2 cuDNN/7.5/CUDA-9.2; \
source /data/HiTIF/progs/miniconda/miniconda2/bin/activate segmentation_model 

current_dir=`pwd`
echo "Running from:$current_dir"

#Add current directory to python path
my_dir=$(cd `dirname $0` && pwd)
export PYTHONPATH=$my_dir:$PYTHONPATH

python -u $my_dir/train_test_unets_fpns.py "$@"
source /data/HiTIF/progs/miniconda/miniconda2/bin/deactivate segmentation_model
