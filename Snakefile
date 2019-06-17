import configparser 
from job_utils import  preprocess_fun,  get_exp


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
    print("Here", configuration)
    if not os.path.exists(conf_file):
        return [configuration]
    else:
        return [ configuration, conf_file]


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

h5_exp_file = "aug_images.h5"
aug_config_file = "config.json"

knime_script = "/data/HiTIF/data/dl_segmentation_paper/knime/launch_scripts/launch_knime37x_workflow.sh"
preprocess_dir = "preprocess"

augment_dir = "augment"
augment_workflow = "/data/HiTIF/data/dl_segmentation_paper/knime/workflows/HiTIF_AugmentInputGT_H5_OutLoc_JSON.knwf"

combine_script="/data/HiTIF/data/dl_segmentation_paper/code/python/combine-all.py"

h5_location = os.path.join(augment_dir, "{exp}", h5_exp_file)
combined_h5 = os.path.join("combined", h5_exp_file)
print(h5_location)
print(combined_h5)

print(h5_location)


rule all:
    input:
        #expand("preprocess/{exp}/config.json", exp=exp_names)
        #expand(h5_location, exp=exp_names)
        combined_h5


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
        preprocess_fun(configuration, exp_tuple, output_json, h5_file)

rule augment:
    input: rules.preprocess.output
    output: 
        h5_location 
    run:
        shell_cmd=knime_script + " " + augment_workflow + " " +  str(input)
        print(shell_cmd)
        shell(shell_cmd)

rule combine_augment:
    input: 
        expand(h5_location, exp=exp_names)
    output:
        combined_h5
    run:    
        combine_cmd = combine_script + " " + input + " " +  output
        print(combine_cmd)
        shell(combine_cmd)
       
