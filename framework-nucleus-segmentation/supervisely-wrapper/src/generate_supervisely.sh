#!/bin/bash

export supervisely_path="/data/HiTIF/data/dl_segmentation_input/utils/supervisely"
src_path=$supervisely_path/src
template_path=$supervisely_path/supervisely_template

export PYTHONPATH=$PYTHONPATH:$src_path
