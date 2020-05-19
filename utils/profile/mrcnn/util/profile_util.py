import configparser
import sys
import re

def replace_word(input_file_path, output_file_path, word_before, word_after):
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


def get_indent(input_file, word):
    with open(input_file) as search:
        for line in search:
            line = line.rstrip()
            if word in line:
                # match = re.search(r'[^A-Za-z]',line)
                r = re.compile(r"^ *")
                match = r.findall(line)[0]
                return match
    return ""


def add_benchmark_on_function(file_name, function_api, function_name):
    # Replace the target string using parser.
    if "(" in function_api:
        input_arg = "".join(("".join(function_api.split('(')[1:])).split(')')[:-1])
    else:
        input_arg = ""

    indent = get_indent(file_name, function_api)

    result_string = '\n' + indent + 'lp = LineProfiler()'
    result_string += '\n' + indent + 'lp_wrapper = lp(' + function_name + ')'
    result_string += '\n' + indent + 'lp_wrapper(' + input_arg + ')'
    result_string += '\n' + indent + 'lp.print_stats()\n'

    replace_word(file_name, file_name, function_api, function_api + result_string)


# Argument Configuration
if len(sys.argv) != 2:
    print("Error: The total number of arguments should be 3.")
    exit()

config_file_path = sys.argv[1]
# config_file_path="00_mrcnn/config.ini"

# Parser (Config file -> Dictionary)
config = configparser.ConfigParser()
config.read(config_file_path)
param = {s: dict(config.items(s)) for s in config.sections()}['general']

# File location configuration
for key in param:
    if key == "iter":
        param[key] = int(param[key])
    else:
        param[key] = (param[key])[1:-1]

demo_input = "demo_profile.py"

# Replace words
replace_word(demo_input, demo_input, "$ITER", str(param["iter"]))
replace_word(demo_input, demo_input, "$IMG_PATH", str(param["img_path"]))
add_benchmark_on_function(param['file_name'], param['function_api'], param['function_name'])
