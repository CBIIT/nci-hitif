#!/bin/bash

source /data/HiTIF/progs/miniconda/miniconda2/bin/activate mrcnn-05-2019
my_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
top_level=$my_dir/../

current_dir=`pwd`
script=${top_level}/Mask_RCNN/samples/cell/cell_script.py
python $script train "$@"
source /data/HiTIF/progs/miniconda/miniconda2/bin/deactivate 
