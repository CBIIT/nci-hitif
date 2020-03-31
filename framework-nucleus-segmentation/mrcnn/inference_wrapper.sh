#!/bin/bash 


if [ "$#" -lt 6 ]
then
    echo "Error: Usage: run_hitif_inference <image> <output_dir> <weights> <crop-size> <padding> <threshold>"
    exit 1
fi

source /data/HiTIF/progs/miniconda/miniconda2/bin/activate mrcnn-05-2019

my_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
top_level=$my_dir/../
inference_script="$top_level"/Mask_RCNN/samples/cell/inference_script.py

image="$1"
output_dir=$2
weights=$3
cropsize=$4
padding=$5
threshold=$6

set -x
filename=$(basename "$image")
echo "$image"
python $inference_script "$image" "$weights" "$output_dir"/"$filename" --cropsize=${cropsize} --padding=${padding} --threshold=${threshold}
