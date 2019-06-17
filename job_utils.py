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
    knime_sec = "generate_augmented_H5_KNIMEWorkflow"
    conf_dir = os.path.abspath(os.path.dirname(output))
    aug_file_name = os.path.join(conf_dir, "augment.cfg")
    
    exp_name, exp_config_path = experiment

    merged_config = configparser.ConfigParser()
    merged_config.optionxform = str
    merged_config.read([exp_config_path, gen_config])

    knime_params = merged_config[knime_sec]
 
    knime_params["imgaugconfigfilepath"] = aug_file_name
    aug_dir, aug_h5 = os.path.split(h5_exp_file)
    knime_params["outDirectoryvar"] = os.path.abspath(aug_dir)
    knime_params["outputh5fname"] = os.path.basename(aug_h5)
    knime_dict = dict(merged_config.items(knime_sec))
    for item in knime_dict.keys():
        clean_string = knime_dict[item].replace('"', "")
        try:
           value = eval(clean_string) 
           knime_dict[item]  = value
        except:
            knime_dict[item] = clean_string
    with open(output, 'w') as json_output:
       json.dump(knime_dict, json_output, indent=2 ) 
    
    #Remove all sections except for aug_sec in config and print it oupt
    #all_sections = merged_config.sections()
    #all_sections.remove(aug_sec)
    #for section in all_sections:
    #    merged_config.remove_section(section)
    with open(aug_file_name, 'w') as configfile:
        merged_config.write(configfile)  


   
