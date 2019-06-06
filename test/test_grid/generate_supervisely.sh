export supervisely_path="../../"
src_path=$supervisely_path/src
template_path=$supervisely_path/supervisely_template

export PYTHONPATH=$PYTHONPATH:$src_path

project_dir=supervisely_project
rm -rf $project_dir 2>/dev/null

set -x 
#cp -r $template_path $project_dir
mkdir $project_dir

fov_file="input.tif"
masks_file="masks.tif"
out_dir="test"

python $src_path/supervisely_grid.py "${fov_file}" "${masks_file}" --out_dir=${project_dir}/${out_dir} --shape=bitmap
