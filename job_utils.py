import os
import configparser
import json

def get_exp(experiment_name, experiments):
    """
    Returns the experiment tuple from the list of experiments

    Arguments:
        experiment_name: str 

        experiments: list
            list of experiemnt tuples

    Returns: tuple
    """

    exp_tuple = [i for i in experiments if i[0] == experiment_name][0]

    return exp_tuple
    

def get_merged_config(general_config, experiment_config):
    """
        Returns the merged configuration file
        
        Arguments:
            general_config: path 
                Path to the general configuration file
            experiment_config: path
                Path to the per experiment configuration file

        Returns
            merged: configparser
    """

    merged_config = configparser.ConfigParser()
    merged_config.optionxform = str
    merged_config.read([general_config, experiment_config])
    return merged_config

def watershed_2_fun(gen_config, 
                   experiment,
                   output_json, 
                   output_cfg, 
                   gaussian_h5,
                   gaussian_json,
                   edt_h5,
                   edt_json,
                   ):
    """
    Generate the knime augmentation json file
    Arguments:
        gen_config: dict
            The general watershed_2 parameters
        experiment: tuple(2)
            experiment_name, and config file
        output: dirpath 
            Path to output directory 
    """
    watershed_2_knime_sec = "GBDMsWS_KNIMEWorkflow"
    general_sec = "general"


    conf_dir = os.path.abspath(os.path.dirname(output_json))
    watershed_2_cfg = os.path.abspath(output_cfg)
    exp_name, exp_config_path = experiment

    merged_config = get_merged_config(gen_config, exp_config_path) 


    general_params = merged_config[general_sec]
    knime_params = merged_config[watershed_2_knime_sec]

    #Set the input files and output directory
    knime_params["inputDirectory"] = os.path.abspath(general_params["input_directory"].replace('"', ''))
    knime_params["inputRegexSelectionStr"] = general_params["input_regex"]
    knime_params["outDirectoryvar"] = os.path.join(conf_dir, "images")

    #Set the deep learning trained models 
    knime_params["deeplearningGBModelJSONFname"] = os.path.abspath(gaussian_json)
    knime_params["deeplearningGBModelFname"] = os.path.abspath(gaussian_h5)
    knime_params["deeplearningDMModelJSONFname"] = os.path.abspath(edt_json )
    knime_params["deeplearningDMModelFname"] = os.path.abspath(edt_h5)

    knime_dict = dict(merged_config.items(watershed_2_knime_sec))
    write_dict_to_knime_json(knime_dict, output_json)
    with open(watershed_2_cfg, 'w') as configfile:
        merged_config.write(configfile)  

def preprocess_fun(gen_config, experiment, output, h5_exp_file="aug_images.h5"):
    """
    Generate the knime augmentation json file
    Arguments:
        gen_config: dict
            The general augmentation parameters
        experiment: tuple(2)
            experiment_name, and config file
        output: filepath
            Path to output json file
    """
    aug_sec = "augmentation"
    general_sec = "general"
    knime_sec = "generate_augmented_H5_KNIMEWorkflow"
    conf_dir = os.path.abspath(os.path.dirname(output))
    aug_file_name = os.path.join(conf_dir, "augment.cfg")
    
    exp_name, exp_config_path = experiment

    merged_config = configparser.ConfigParser()
    merged_config.optionxform = str
    merged_config.read([gen_config, exp_config_path])

    
    knime_params = merged_config[knime_sec]
    general_params = merged_config[general_sec]

    #Convert the image regex and directories to Knime params
    knime_params["greyimagesrootInputFolder"]  = general_params["input_directory"]
    knime_params["gtimagesrootInputFolder"] = general_params["ground_truth_directory"]
    knime_params["gtregexFileSelectionStr"] = general_params["input_regex"]
    knime_params["greyregexFileSelectionStr"] = general_params["input_regex"]
    knime_params["imgaugconfigfilepath"] = aug_file_name
    aug_dir, aug_h5 = os.path.split(h5_exp_file)
    knime_params["outDirectoryvar"] = os.path.abspath(aug_dir)
    knime_params["outputh5fname"] = os.path.basename(aug_h5)
    knime_dict = dict(merged_config.items(knime_sec))
    write_dict_to_knime_json(knime_dict, output)
    with open(aug_file_name, 'w') as configfile:
        merged_config.write(configfile)  

def write_dict_to_knime_json(dictionary, json_output):
    """
    Write out a python dictionary in a json format accepted
    by Knime

    Arguments:
        dictionary: dict
            The python dictionary
        json_path: str
            The path of the json file 
    """
    knime_dict = dictionary 
    for item in knime_dict.keys():
        clean_string = knime_dict[item].replace('"', "")
        try:
           value = eval(clean_string) 
           knime_dict[item]  = value
        except:
            knime_dict[item] = clean_string
    with open(json_output, 'w') as json_file:
       json.dump(knime_dict, json_file, indent=2 ) 
    

def train_unet_fpn(work_dir, conf, h5_in, model_h5, model_json):
    """
    Train then unet/fpn models
    Arguments:
        work_dir:  path
            the directory where training will take place
        conf: file_path
            the input configuration file
        h5_in: file_path
            The input and ground truth 
        model_h5: file_path
            The location of the trained model weights
        output_h5: file_path
            The location of the trained model architecture
    """
    
    h5_in = os.path.abspath(h5_in) 
    model_h5 = os.path.abspath(model_h5) 
    model_json = os.path.abspath(model_json) 
    conf = os.path.abspath(conf)
    top_dir = os.getcwd()
    os.chdir(work_dir)
    train_unet_fpn_script = "/data/HiTIF/data/dl_segmentation_paper/code/python/unet_fpn/train_model.sh"
    #config_file = os.path.join(work_dir, dl_config)
    #os.system("cp {0} {1}".format(conf, config_file))
    shell_cmd = train_unet_fpn_script + \
       " train " + \
       " --h5fname " +   h5_in +  \
       " --trained_h5 " + model_h5 + \
       " --trained_json " + model_json + \
       " -c " + conf

    print(shell_cmd)
    os.system(shell_cmd)
    os.chdir(top_dir)
   
def maps_fun(      
        merged_config, 
        inference_dir,
        output_file, 
        ):
    """
    Generate the knime maps workflow json file
    Arguments:
        merged_config: config
            The merged inference configuration file 
        inference_dir: str 
            path to the inference results 
        output_file: str 
            Path to maps output 
    """
    map_knime_sec = "calculate_mAP_KNIMEWorkflow"
    general_sec = "general"
    general_params = merged_config[general_sec]
    knime_params = merged_config[map_knime_sec]

    out_dir, out_file = os.path.split(os.path.abspath(output_file))
    #Set the images directories and regex 
    knime_params["greyimagesrootInputFolder"] = general_params["input_directory"]
    knime_params["greyregexFileSelectionStr"] = general_params["input_regex"]

    knime_params["gtimagesrootInputFolder"] = general_params["ground_truth_directory"]
    knime_params["gtregexFileSelectionStr"] = general_params["input_regex"]

    knime_params["inferimagesrootInputFolder"] = inference_dir 
    knime_params["inferregexFileSelectionStr"] = general_params["input_regex"]

    knime_params["outDirectoryvar"] = out_dir
    knime_params["csvoutputprefix"] = out_file

    knime_json_output = os.path.join(out_dir,"knime.json" )

    knime_dict = dict(merged_config.items(map_knime_sec))
    write_dict_to_knime_json(knime_dict, knime_json_output)
    merged_file_path = os.path.join(out_dir,"map.cfg")
    with open(merged_file_path, 'w') as configfile:
        merged_config.write(configfile)  

    return knime_json_output


