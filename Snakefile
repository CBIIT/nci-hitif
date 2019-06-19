import configparser 
from job_utils import  preprocess_fun,  get_exp, train_unet_fpn


def get_exp_configurations(wildcards):
    """ 
    Returns the configuration file related to a given experiment
    Arguments:
        wildcards: str
            The snakemake value for the wildcard for a given rule 
    """
    print(wildcards.exp)
    exp_tuple = get_exp(wildcards.exp, input_exp) 
    conf_file = exp_tuple[1]
    if not os.path.exists(conf_file):
        return [gen_aug_config]
    else:
        return [gen_aug_config, conf_file]


my_config = configparser.ConfigParser()
configuration = os.path.abspath(config["conf"])
my_config.read(configuration)

#Find all the input experiment names
input_exp = eval(my_config["general"]["experiments"])
exp_names = [exp[0] for exp in input_exp]
exp_config = [exp[1] for exp in input_exp if os.path.exists(exp[1])]

#Find the work dir
pipeline_dir = os.path.abspath(my_config["general"]["workdir"])
#if not os.path.isdir(pipeline_dir):
workdir: my_config["general"]["workdir"]

configs_dir =  my_config["general"]["configs_dir"]

#preprocess
gen_aug_config = os.path.join(configs_dir, "imgaug.cfg") 

h5_exp_file = "aug_images.h5"
aug_config_file = "config.json"

knime_script = "/data/HiTIF/data/dl_segmentation_paper/knime/launch_scripts/launch_knime37x_workflow.sh"
preprocess_dir = "preprocess"

#Augment
augment_dir = "augment"
augment_workflow = "/data/HiTIF/data/dl_segmentation_paper/knime/workflows/HiTIF_AugmentInputGT_H5_OutLoc_JSON.knwf"

#Combine
combine_script="python /data/HiTIF/data/dl_segmentation_paper/code/python/combine-all.py"
h5_location = os.path.join(augment_dir, "{exp}", h5_exp_file)
combined_h5 = os.path.join("combined", h5_exp_file)


#DL commands
train_dir = "train"
#That file should not change
dl_config = "my_config.cfg"

train_unet_fpn_script = "/data/HiTIF/data/dl_segmentation_paper/code/python/unet_fpn/train_model.sh"

#Gaussian training
gaussian_config = os.path.join(configs_dir, "gaussian-config.cfg") 
train_gaussian_dir = os.path.join(train_dir, "gaussian")

#edt training
edt_config = os.path.join(configs_dir, "edt-config.cfg") 
train_edt_dir = os.path.join(train_dir, "edt")

#mask training
mask_config = os.path.join(configs_dir, "mask-config.cfg") 
train_mask_dir = os.path.join(train_dir, "mask")

#outline training
outline_config = os.path.join(configs_dir, "outline-config.cfg") 
train_outline_dir = os.path.join(train_dir, "outline")

#mrcnn training
train_mrcnn= "/data/HiTIF/data/dl_segmentation_paper/code/python/mask-rcnn/mask-rcnn-latest/train/train-wrapper.sh"
mrcnn_config = os.path.join(configs_dir, "mrcnn-config.cfg") 
train_mrcnn_dir = os.path.join(train_dir, "mrcnn")

rule all: 
    input:
        #expand("preprocess/{exp}/config.json", exp=exp_names)
        #expand(h5_location, exp=exp_names)
        combined_h5
        #rules.train_outline.output.h5
        #rules.train_mask.output.h5,
        #rules.train_edt.output.h5,
        #rules.train_gaussian.output.h5

rule preprocess:
    input:
        get_exp_configurations
    output:
        preprocess_dir + "/{exp}/" + aug_config_file
    run:
        exp_name = wildcards.exp
        exp_tuple = get_exp(exp_name, input_exp)
        output_json = str(output)
        h5_file = h5_location.replace("{exp}", exp_name)
        preprocess_fun(gen_aug_config, exp_tuple, output_json, h5_file)

rule augment:
    input: rules.preprocess.output
    output: 
        h5_location 
    run:
        json_file = os.path.abspath(str(input))
        shell_cmd=knime_script + " " + augment_workflow + " " +  str(input)
        print(shell_cmd)
        shell(shell_cmd)

rule combine_augment:
    input: 
        expand(h5_location, exp=exp_names)
    output:
        combined_h5
    run:    
        combine_cmd = combine_script + " " + str(input) + " " +  str(output)
        print(combine_cmd)
        shell(combine_cmd)
      
rule train_gaussian:
    input:
        h5 = combined_h5,
        cfg = gaussian_config
    output:
        h5 = os.path.join(train_gaussian_dir, "trained.h5"),
        json = os.path.join(train_gaussian_dir,"trained.json")
    run:
        train_unet_fpn(train_gaussian_dir, input.cfg, input.h5, output.h5, output.json)

rule train_edt:
    input:
        h5 = combined_h5,
        cfg = edt_config
    output:
        h5 = os.path.join(train_edt_dir, "trained.h5"),
        json = os.path.join(train_edt_dir,"trained.json")
    run:
        train_unet_fpn(train_edt_dir, input.cfg, input.h5,output.h5, output.json)


rule train_mask:
    input:
        h5 = combined_h5,
        cfg = mask_config
    output:
        h5 = os.path.join(train_mask_dir, "trained.h5"),
        json = os.path.join(train_mask_dir,"trained.json")
    run:
        train_unet_fpn(train_mask_dir, input.cfg, input.h5,output.h5, output.json)

rule train_outline:
    input:
        h5 = combined_h5,
        cfg = outline_config
    output:
        h5 = os.path.join(train_outline_dir, "trained.h5"),
        json = os.path.join(train_outline_dir,"trained.json")
    run:
        train_unet_fpn(train_outline_dir, input.cfg, input.h5,output.h5, output.json)

rule train_mrcnn:
    input:
        h5 = combined_h5,
        cfg = mrcnn_config
    output:
        model_h5 = os.path.join(train_mrcnn_dir, "last.h5"),
    run:
         
        #dl_config = "my_config.cfg"
        #config_file = os.path.join(train_mrcnn_dir, dl_config)
        #os.system("cp {0} {1}".format(input.cfg, config_file))
        cmd = train_mrcnn +  \
            " --dataset "  + input.h5 +  \
            " --logs " +  train_mrcnn_dir + \
            " --latest  "  + output.model_h5 + \ 
            " -c " + input.cfg
        print(cmd)
        shell(cmd)
