import configparser 
from job_utils import  preprocess_fun,get_exp,train_unet_fpn,watershed_2_fun, get_merged_config, maps_fun
from shutil import copyfile

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

    Returns:
        [general_configuration]
            If experiment does not have a specific configuration file
        [general_configuration, exoperiment_configuration]
            If experiment has a specific configuration file
    """
    #print(wildcards.exp)
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
    #print(wildcards.exp)
    return get_exp_configurations(wildcards, gen_aug_config, input_exp)


def get_inference_exp_configurations(wildcards):
    """ 
    Returns the configuration file related to a given experiment
    Arguments:
        wildcards: str
            The snakemake value for the wildcard for a given rule 
    """
    #print(wildcards.exp)
    return get_exp_configurations(wildcards, inference_config, inference_exp)


def get_inference_images_paths(exp_name, folder):
    """
    Returns a list of images names of an ground truth experiment
    Arguemnts:
        exp_name: str
            The ground truth experiment name mentioned in global config 
        folder: str
            Can either be "ground_truth" or "gray_scale" 

    Returns: list
    """

    exp_name, exp_config = get_exp(exp_name, inference_exp)
    merged_config = get_merged_config(inference_config, exp_config)
    general_sec = "general" 
    if folder == "ground_truth":
        dir_attribute = "ground_truth_directory"
    elif folder == "gray_scale":
        dir_attribute = "input_directory"
    else:
        raise Exception("The folder value:{0} is incorrect".format(folder)) 

    #print(merged_config[general_sec])
    input_dir = merged_config[general_sec][dir_attribute].replace('"','')
    input_regex = merged_config[general_sec]["input_regex"].replace('"','')

    import glob
    joined_regex = os.path.join(input_dir, input_regex)
    #print("{0} Regex:".format(folder), joined_regex)
    input_files = glob.glob(joined_regex)
    #print("{0} Input files".format(folder), input_files)
    return input_files
 

def get_inference_gray_images(wildcards):
    """
    Returns a list of full path of the inference images per experiment
    Arguemnts:
        wildcards: contains "exp"
            The inference experiment name

    Returns: list
    """
    exp_name = wildcards.exp
    return get_inference_images_paths(exp_name, "gray_scale")


def get_inference_ground_truth_images(wildcards):
    """
    Returns a list of full path of the ground truth images per experiment
    Arguments:
        wildcards: contains "exp"
            The inference experiment name

    Returns: list
    """
    exp_name = wildcards.exp
    return get_inference_images_paths(exp_name, "ground_truth")


def get_method_inference_dir(method_name):
    """
    Returns the inference directory defined in the method 
    Arguments:
        method_name: str
            The method can point to any of the three techniques["watershed-2", "watershed-4", "mrcnn"]
    """

    if method_name == "watershed-2":
        inference_dir = rules.watershed_2_execute.output.out_dir
    elif method_name == "mrcnn":
        inference_dir = rules.infer_mrcnn.output.out_dir
    else:
        raise Exception("Method {0} not implemented".format(method_name))

    return inference_dir

def get_wildcard_method_inf_dir(wildcards):
    method_name = wildcards.method
    return get_method_inference_dir(method_name)

def get_map_images(wildcards):
    """
    Returns a tuple of two lists of the all images required to calculate maps:
        a- The ground truth images
        b- The inference results

    Arguments:
        wildcards: contains "exp", and "method"
        The method can point to any of the three techniques["watershed-2", "watershed-4", "mrcnn"]
    """
    exp_name = wildcards.exp
    method_name = wildcards.method
    ground_truth_files = get_inference_images_paths(exp_name, "ground_truth")

    inference_dir = get_method_inference_dir(method_name)
    #Get the image names
    image_names = [os.path.basename(image) for image in ground_truth_files]
    inference_results =[os.path.join(inference_dir,image) for image in image_names]
    #print(inference_results)
    #return ground_truth_files + inference_results
    return ground_truth_files #+ inference_results


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

#Create a log dir
log_dir = os.path.join(pipeline_dir, "log")
if not os.path.isdir(log_dir):
    os.mkdir(log_dir)

#preprocess
gen_aug_config = os.path.join(configs_dir, "imgaug.cfg") 

h5_exp_file = "aug_images.h5"
aug_config_file = "config.json"

knime_script = "/data/HiTIF/data/dl_segmentation_paper/knime/launch_scripts/launch_knime37x_workflow.sh"
preprocess_dir = "preprocess"

#Augment
augment_dir = "augment"
augment_workflow = "/data/HiTIF/data/dl_segmentation_paper/knime/workflows/HiTIF_AugmentInputGT_H5_OutLoc_JSON.knwf"
augment_log = os.path.join(augment_dir, "{exp}", "augment.log")

#Combine
combine_script="python /data/HiTIF/data/dl_segmentation_paper/code/python/combine-all.py"
h5_location = os.path.join(augment_dir, "{exp}", h5_exp_file)
combined_h5 = os.path.join("combined", h5_exp_file)
combined_log = os.path.join("combined", "combine.log")


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
model_h5_file = os.path.join(train_mrcnn_dir, "last.h5")
mrcnn_train_log = os.path.join(train_mrcnn_dir, "mrcnn_train.log")


#inference

inference_exp = eval(my_config["general"]["infer_experiments"])
inference_exp_names = [exp[0] for exp in inference_exp]
inference_dir = "inference"
inference_config = os.path.join(configs_dir, "inference-config.cfg") 


#inference preparation
exp_merged_config = os.path.join(inference_dir, "preprocess", "{exp}", "config.cfg")


#watershed_2 inference
watershed_2_loc = os.path.join(inference_dir, "watershed-2") 
watershed_2_exp = os.path.join(watershed_2_loc, "{exp}")

#Watershed outputs
watershed_2_knime_json = os.path.join(watershed_2_exp, "knime.json")
watershed_2_cfg = os.path.join(watershed_2_exp, "config.cfg")
watershed_2_images_dir = os.path.join(watershed_2_exp,"images")
#print("DIR:{0}".format(watershed_2_images_dir ))


#watershed_2_workflow = "/data/HiTIF/data/dl_segmentation_paper/knime/workflows/HiTIF_CV7000_Nucleus_Segmentation_DeepLearning_IncResV2FPN_GBDMsWS_nonSLURM_37X_OutLoc_JSON.knwf"
watershed_2_workflow = "/data/HiTIF/data/dl_segmentation_paper/knime/workflows/HiTIF_CV7000_Nucleus_Segmentation_DeepLearning_IncResV2FPN_watershed2_serial.knwf"


#mrcnn_infernce
mrcnn_images_dir=os.path.join(inference_dir, "mrcnn","{exp}")
#infer_mrcnn="/gpfs/gsfs11/users/zakigf/mask-rcnn-with-augmented-images/Mask_RCNN/images/cell-images/inference/hitif_ml_segmentation/utils/run_hitif_inference.sh"
infer_mrcnn="/data/HiTIF/data/dl_segmentation_paper/code/python/mask-rcnn/mask-rcnn-latest/inference/run_hitif_inference.sh"
mrcnn_infer_log = os.path.join(inference_dir, "mrcnn", "mrcnn_inference-{exp}.log")

#Mean Average Precision

maps_dir = "mean-average-precision"
#Define a method widlcard that can point to ["watershed-2", "watershed-4", "mrcnn"] 
maps_exp_dir = os.path.join(maps_dir, "{method}", "{exp}")
maps_output_file = os.path.join(maps_exp_dir,"maps.csv")
maps_workflow = "/data/HiTIF/data/dl_segmentation_paper/knime/workflows/HiTIF_Calculate_mAP_GTvsInference_Python_3Inputs_OutLoc_JSON.knwf"


#define local rules that will not be submitted to the cluster
localrules: all, preprocess, combine_augment, inference_prep, watershed_2_prep

rule all: 
    input:
#        pass
#        #expand("preprocess/{exp}/config.json", exp=input_exp_names)
#        #expand(h5_location, exp=input_exp_names)
#         expand(watershed_2_knime_json, exp=inference_exp_names)
#         expand(os.path.join(watershed_2_exp, "images"), exp=inference_exp_names)
         #expand(maps_output_file, exp=inference_exp_names, method = ["mrcnn"]),
         expand(maps_output_file, exp=inference_exp_names, method = ["watershed-2", "mrcnn"]),
         #expand(exp_merged_config, exp = inference_exp_names) 

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
    log:
        preprocess_dir + "/{exp}/" + "log.log"
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
    log:
        augment_log   
    run:
        json_file = os.path.abspath(str(input))
        shell_cmd = knime_script + " " + augment_workflow + " " +  str(input) + " &> " + str(log)
        print(shell_cmd)
        shell(shell_cmd)

rule combine_augment:
    input: 
        expand(h5_location, exp=input_exp_names)
    output:
        combined_h5
    log:
        combined_log
    run:    
        combine_cmd = combine_script + " " + str(input) + " " +  str(output) + " &> " + str(log)
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


rule inference_prep:
    input:
        exp_config = get_inference_exp_configurations
    output:
        exp_merged_config
    run:
        if len(input.exp_config) == 1:
            copyfile(str(input.exp_config[0]), str(output))
        else:
            exp_config = get_merged_config(str(input.exp_config[0]),str(input.exp_config[1])) 
            with open(str(output), 'w') as configfile:
                exp_config.write(configfile)  

rule watershed_2_prep:
    input:
        gaussian_h5 = rules.train_gaussian.output.h5,
        gaussian_json = rules.train_gaussian.output.json,

        edt_h5 = rules.train_edt.output.h5,
        edt_json = rules.train_edt.output.json,

        #The prep is based on the configuration files
        exp_config = get_inference_exp_configurations
    output:    
        knime_json = watershed_2_knime_json, 
        config = watershed_2_cfg
    run:

        exp_name = wildcards.exp.replace("/", "")
        exp_tuple = get_exp(exp_name, inference_exp)
        #Generate the combined configuration file

        watershed_2_fun(inference_config,  
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
        images = get_inference_gray_images,
        knime_json = rules.watershed_2_prep.output.knime_json,
        conf = rules.watershed_2_prep.output.config
    output: 
        out_dir = directory(watershed_2_images_dir),
        #images = os.path.join(watershed_2_images_dir,"AUTO0496_N14_T0001F002L01A01Z01C01.tif")
        #images = os.path.join(watershed_2_images_dir,"{image}.tif")

    run:
        #print("Here", str(input.images))
        #print(str(output))
        shell_cmd = knime_script + " " + watershed_2_workflow + " " + os.path.abspath(input.knime_json)
        print(shell_cmd)
        shell(shell_cmd)


rule map_execute:
    input:
       images = get_map_images,
       inf_dir = get_wildcard_method_inf_dir
    output:
        #out_dir = directory(maps_exp_dir) 
        out_file = maps_output_file
    run:
        #print("MAPS:", str(input.images))
        #print("MAPS:", str(output.out_file))
        exp_name = wildcards.exp.replace("/", "")
        exp_name, exp_config_path = get_exp(exp_name, inference_exp)
        merged_config = get_merged_config(inference_config, exp_config_path ) 
        #Get the inference directory for the method 
        method_name = wildcards.method.replace("/", "")
        inference_dir = os.path.abspath(get_method_inference_dir(method_name).replace("{exp}", exp_name))
        knime_json = maps_fun(merged_config,inference_dir,str(output.out_file))

        shell_cmd = knime_script + " " + maps_workflow + " " + os.path.abspath(knime_json)
        print(shell_cmd)
        shell(shell_cmd)

rule train_mrcnn:
    input:
        h5 = combined_h5,
        cfg = mrcnn_config
    output:
        model_h5 = model_h5_file
    log:
        mrcnn_train_log
    run:
         
        #dl_config = "my_config.cfg"
        #config_file = os.path.join(train_mrcnn_dir, dl_config)
        #os.system("cp {0} {1}".format(input.cfg, config_file))
        cmd = train_mrcnn +  \
            " --dataset "  + input.h5 +  \
            " --logs " +  train_mrcnn_dir + \
            " --latest  "  + output.model_h5 + \ 
            " -c " + input.cfg \
            + " &>> {0}".format(str(log))
        print(cmd)
        shell(cmd)


rule infer_mrcnn:
    input:
        model_h5 = rules.train_mrcnn.output.model_h5, 
        images = get_inference_gray_images,
        config = rules.inference_prep.output
    output:
        out_dir = directory(mrcnn_images_dir)
    log: 
        mrcnn_infer_log
    run:

        #Get the inference parameters
        config = configparser.ConfigParser()
        config.optionxform = str
        config.read(input.config)
        mrcnn_sec = "mrcnn"
        mrcnn_params = config[mrcnn_sec] 

        threshold = config.getint(mrcnn_sec, "threshold")
        cropsize = config.getint(mrcnn_sec, "cropsize")
        padding = config.getint(mrcnn_sec, "padding")

        os.mkdir(output.out_dir)
        for image in input.images:
            cmd = infer_mrcnn +  \
                ' "{0}" "{1}" "{2}" "{3}" "{4}" "{5}" &>> {6}'.format( \
                os.path.abspath(image), \
                os.path.abspath(str(output.out_dir)), \
                os.path.abspath(str(input.model_h5)), \
                cropsize, \
                padding, \
                threshold,
                str(log))
            print(cmd)
            shell(cmd)

