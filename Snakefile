import configparser 
from job_utils import  preprocess_fun,get_exp,train_unet_fpn,watershed_2_fun, get_merged_config


def get_exp_configurations(wildcards, general_config,experiments):
    """ 
    Returns the configuration file related to a given experiments
    Arguments:
        wildcards: str
            The snakemake value for the wildcard for a given rule 
        general_config: path 
            The general configuration file for the rule
        experiments: list(tuples)
            The list 
    """
    print(wildcards.exp)
    #If exp is a directory, remove the forward slash
    
    experiment_name = wildcards.exp.replace("/", "")
    exp_tuple = get_exp(experiment_name, experiments) 
    conf_file = exp_tuple[1]
    if not os.path.exists(conf_file):
        return [general_config]
    else:
        return [general_config, conf_file]


def get_input_exp_configurations(wildcards):
    """ 
    Returns the configuration file related to a given experiment
    Arguments:
        wildcards: str
            The snakemake value for the wildcard for a given rule 
    """
    print(wildcards.exp)
    return get_exp_configurations(wildcards, gen_aug_config, input_exp)


def get_watershed_2_exp_configurations(wildcards):
    """ 
    Returns the configuration file related to a given experiment
    Arguments:
        wildcards: str
            The snakemake value for the wildcard for a given rule 
    """
    print(wildcards.exp)
    return get_exp_configurations(wildcards, watershed_2_config, inference_exp)
   

def get_inference_images(wildcards):
    exp_name = wildcards.exp
    exp_name, exp_config = get_exp(exp_name, inference_exp)
    merged_config = get_merged_config(watershed_2_config, exp_config)
    knime_sec = "GBDMsWS_KNIMEWorkflow" 
    input_dir = merged_config[knime_sec]["inputDirectory"].replace('"','')
    input_regex = merged_config[knime_sec]["inputRegexSelectionStr"].replace('"','')
    import glob
    joined_regex = os.path.join(input_dir, input_regex)
    print(joined_regex)
    input_files = glob.glob(joined_regex)
    print(input_files)
    input_pathes = [os.path.join(watershed_2_loc, exp_name, tif)
        for tif in input_files]
    print(input_pathes)
    return input_files


my_config = configparser.ConfigParser()
configuration = os.path.abspath(config["conf"])
my_config.read(configuration)

#Find all the input experiment names
input_exp = eval(my_config["general"]["experiments"])
input_exp_names = [exp[0] for exp in input_exp]
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


#inference

inference_exp = eval(my_config["general"]["infer_experiments"])
inference_exp_names = [exp[0] for exp in inference_exp]
inference_dir = "inference"

#watershed_2 inference

watershed_2_config = os.path.join(configs_dir, "watershed-2-config.cfg") 
watershed_2_loc = os.path.join(inference_dir, "watershed-2") 
watershed_2_exp = os.path.join(watershed_2_loc, "{exp}")
watershed_2_knime_json = os.path.join(watershed_2_exp, "knime.json")
watershed_2_cfg = os.path.join(watershed_2_exp, "config.cfg")

watershed_2_workflow = "/data/HiTIF/data/dl_segmentation_paper/knime/workflows/HiTIF_CV7000_Nucleus_Segmentation_DeepLearning_IncResV2FPN_GBDMsWS_nonSLURM_37X_OutLoc_JSON.knwf"


rule all: 
    input:
#        pass
#        #expand("preprocess/{exp}/config.json", exp=input_exp_names)
#        #expand(h5_location, exp=input_exp_names)
#         expand(watershed_2_knime_json, exp=inference_exp_names)
         expand(os.path.join(watershed_2_exp, "images"), exp=inference_exp_names)
#        #combined_h5
#        #rules.train_outline.output.h5
#        #rules.train_mask.output.h5,
#        #rules.train_edt.output.h5,
#        #rules.train_gaussian.output.h5

rule preprocess:
    input:
        get_input_exp_configurations
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
        shell_cmd = knime_script + " " + augment_workflow + " " +  str(input)
        print(shell_cmd)
        shell(shell_cmd)

rule combine_augment:
    input: 
        expand(h5_location, exp=input_exp_names)
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
        print(os.path.abspath(output.json))
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

rule watershed_2_prep:
    input:
        gaussian_h5 = rules.train_gaussian.output.h5,
        gaussian_json = rules.train_gaussian.output.json,

        edt_h5 = rules.train_edt.output.h5,
        edt_json = rules.train_edt.output.json,

        exp_directories = get_watershed_2_exp_configurations
    output:    
        knime_json = watershed_2_knime_json, 
        config = watershed_2_cfg
    run:

        exp_name = wildcards.exp.replace("/", "")
        exp_tuple = get_exp(exp_name, inference_exp)
        #Generate the combined configuration file

        watershed_2_fun(watershed_2_config,  
                        exp_tuple,
                        str(output.knime_json), 
                        str(output.config), 
                        str(input.gaussian_h5),
                        str(input.gaussian_json),
                        str(input.edt_h5),
                        str(input.edt_json)
                        )

rule watershed_2_execute:
    input:
        images = get_inference_images,
        knime_json = rules.watershed_2_prep.output.knime_json,
        conf = rules.watershed_2_prep.output.config
    output: 
        directory(os.path.join(watershed_2_exp,"images"))
    run:
        print(str(output))
        shell_cmd = knime_script + " " + watershed_2_workflow + " " + input.knime_json
        print(shell_cmd)
        shell(shell_cmd)

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
