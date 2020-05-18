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

# Argument Configuration
if len(sys.argv)!=2:
    print("Error: The total number of arguments should be 3.")
    exit()
config_file_path = sys.argv[1]

# Parser (Config file -> Dictionary)
config = configparser.ConfigParser()
config.read(config_file_path)
param = {s: dict(config.items(s)) for s in config.sections()}['general']

# File location configuration
for key in param:
    if key == "num_image":
        param[key] = int(param[key])
    else:
         param[key] = True if param[key]== "True" else False

input_dir = "../util/input/"
demo_input = input_dir + "demo_profile.py"
mrcnn_infer_input = input_dir + "mrcnn_infer_profile.py"
inference_input = input_dir + "inference_profile.py"

# Replace words
replace_word(demo_input,"./demo_profile.py","$NUM_ITER",str(param["num_image"]))
if param['mrcnn_infer_profile'] == True:
    replace_word("./demo_profile.py", "./demo_profile.py", "mask = mrcnn_infer(img, mrcnn_model_path, config_file_path)",
                 "lp = LineProfiler()\nlp_wrapper = lp(mrcnn_infer)\nlp_wrapper(img, mrcnn_model_path, config_file_path)\nlp.print_stats()")

if param['stitched_inference_profile'] == True:
    replace_word(mrcnn_infer_input, "./mrcnn_infer_profile.py", "stitched_inference_stack, num_times_visited = stitched_inference(image, cropsize, model, padding=padding)",
                 "lp = LineProfiler()\nlp_wrapper = lp(mrcnn_infer)\nlp_wrapper(img, mrcnn_model_path, config_file_path)\nlp.print_stats()")
else:
    replace_word(inference_input, "./inference_profile.py", "$PROF_STITCHED_INFERENCE", "")

if param['merge_cells_profile'] == True:
    replace_word("./inference_profile.py", "./inference_profile.py", "$PROF_MERGE_CELLS", "@profile")
else:
    replace_word("./inference_profile.py", "./inference_profile.py", "$PROF_MERGE_CELLS", "")

if param['run_inference_profile'] == True:
    replace_word("./inference_profile.py", "./inference_profile.py", "$PROF_RUN_INFERENCE", "@profile")
else:
    replace_word("./inference_profile.py", "./inference_profile.py", "$PROF_RUN_INFERENCE", "")
