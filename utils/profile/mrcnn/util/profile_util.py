import configparser
import sys

def replace_word(input_file_path,output_file_path,word_before,word_after):
    # Read in the file
    f = open(input_file_path, 'r')
    filedata = f.read()
    f.close()

    # Replace the target string
    newdata = filedata.replace(word_before, word_after)

    # Write the file out again
    f = open(output_file_path, 'w')
    f.write(newdata)
    f.close()

def add_benchmark_on_function(input_file_path,output_file_path,function_api,function_name):
    # Replace the target string using parser.
    # function_api = "mask = mrcnn_infer(img, mrcnn_model_path, config_file_path)"
    # function_name = "mrcnn_infer"
    if "(" in function_api:
        input_arg = "".join(("".join(function_api.split('(')[1:])).split(')')[:-1])
    else:
        input_arg = ""

    result_string = '\nlp = LineProfiler()\nlp_wrapper = lp('+function_name+')\nlp_wrapper('+input_arg+')\nlp.print_stats()\n'

    replace_word(input_file_path, output_file_path,function_api,
                 function_api+result_string)

# Argument Configuration
if len(sys.argv)!=2:
    print("Error: The total number of arguments should be 3.")
    exit()
config_file_path = sys.argv[1]
# config_file_path="01_stitched_inference/config.ini"

# Parser (Config file -> Dictionary)
config = configparser.ConfigParser()
config.read(config_file_path)
param = {s: dict(config.items(s)) for s in config.sections()}['general']

# File location configuration
for key in param:
    if key == "num_image":
        param[key] = int(param[key])
    elif key == "function_api":
        param[key] = param[key]
    elif key == "function_name":
        param[key] = param[key]
    else:
         param[key] = True if param[key]== "True" else False

input_dir = "../util/input/"
demo_input = input_dir + "demo_profile.py"
mrcnn_infer_input = input_dir + "mrcnn_infer_profile.py"
inference_input = input_dir + "inference_profile.py"

# Replace words
replace_word(demo_input,"./demo_profile.py","$NUM_ITER",str(param["num_image"]))
if param['mrcnn_infer_profile'] == True:
    # add_benchmark_on_function("./demo_profile.py", "./demo_profile.py",  "mask = mrcnn_infer(img, mrcnn_model_path, config_file_path)","mrcnn_infer")
    add_benchmark_on_function("./demo_profile.py", "./demo_profile.py",param['function_api'], param['function_name'])
    print(param['function_api'])
    # replace_word("./demo_profile.py", "./demo_profile.py", "mask = mrcnn_infer(img, mrcnn_model_path, config_file_path)",
    #              "lp = LineProfiler()\nlp_wrapper = lp(mrcnn_infer)\nlp_wrapper(img, mrcnn_model_path, config_file_path)\nlp.print_stats()")

if param['stitched_inference_profile'] == True:
    replace_word(mrcnn_infer_input, "./mrcnn_infer_profile.py", "stitched_inference_stack, num_times_visited = stitched_inference(image, cropsize, model, padding=padding)",
                 "lp = LineProfiler()\nlp_wrapper = lp(mrcnn_infer)\nlp_wrapper(img, mrcnn_model_path, config_file_path)\nlp.print_stats()")
else:
    replace_word(mrcnn_infer_input, "./mrcnn_infer_profile.py", "$PROF_STITCHED_INFERENCE", "")

if param['merge_cells_profile'] == True:
    replace_word("./inference_profile.py", "./inference_profile.py", "$PROF_MERGE_CELLS", "@profile")
else:
    replace_word("./inference_profile.py", "./inference_profile.py", "$PROF_MERGE_CELLS", "")

if param['run_inference_profile'] == True:
    replace_word("./inference_profile.py", "./inference_profile.py", "$PROF_RUN_INFERENCE", "@profile")
else:
    replace_word("./inference_profile.py", "./inference_profile.py", "$PROF_RUN_INFERENCE", "")
