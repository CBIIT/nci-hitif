# Execution Time profiler
This is an inference execution time profiler for our exisiting demos. **Currently, this only works on NIH biowulf Linux system.** If you want to use them on your own Linux system, please update the part for activating conda environment in driver script.

## Input example:
```bash
ITER = 1
IMG_PATH="../../../visualization/GreyScale/BABE_Biological/Plate1_E03_T0001FF001Zall.tif"
FILE_NAME = "demo_profile.py"
FUNCTION_API = "mask = mrcnn_infer(img, mrcnn_model_path, config_file_path)"
FUNCTION_NAME = "mrcnn_infer"
```
## Output example:
```bash
Line #      Hits         Time  Per Hit   % Time  Line Contents
==============================================================
    12                                           def mrcnn_infer(img, mrcnn_model_path, config_file_path):
    13                                               # Config File Parser
    14         1        355.0    355.0      0.0      config = configparser.ConfigParser()
    15         1        699.0    699.0      0.0      config.read(config_file_path)
    16         1        146.0    146.0      0.0      param = {s: dict(config.items(s)) for s in config.sections()}['general']
    17         4          5.0      1.2      0.0      for key in param:
    18         3          7.0      2.3      0.0          param[key] = int(param[key])
    19         1          1.0      1.0      0.0      cropsize = param['cropsize']
    20         1          1.0      1.0      0.0      padding = param['padding']
    21         1          1.0      1.0      0.0      threshold = param['threshold']
    22                                           
    23         1   21127759.0 21127759.0     23.0      model = generate_inference_model(mrcnn_model_path, cropsize)
    24         1       1499.0   1499.0      0.0      mask = np.zeros(img.shape)
    25                                           
    26         2         10.0      5.0      0.0      for i in range(len(img)):
    27         1       3390.0   3390.0      0.0          imarray = np.array(img[i]).astype('uint16')
    28         1       2677.0   2677.0      0.0          image = skimage.color.gray2rgb(imarray)
    29         1      26706.0  26706.0      0.0          image = map_uint16_to_uint8(image)
    30                                           
    31         1   38342666.0 38342666.0     41.7          stitched_inference_stack, num_times_visited = stitched_inference(image, cropsize, model, padding=padding)
    32                                           
    33         1        395.0    395.0      0.0          masks = CleanMask(stitched_inference_stack, threshold, )
    34                                                   # masks.merge_cells()
    35         1   32444024.0 32444024.0     35.3          n_conn_comp, graph_labels = merge_cells(masks)
    36                                           
    37         1        814.0    814.0      0.0          my_mask = masks.getMasks().astype("int16")
    38         1       2843.0   2843.0      0.0          mask[i, :, :] = my_mask
    39         1          1.0      1.0      0.0      return mask
```

## Prebuilt Test Organization
### MRCNN
1. mrcnn/00_mrcnn:
This profiles **mrcnn_infer** function used in **demo.py**.

2. mrcnn/01_stitched_inference:
This profiles **stitched_inference** function used in **mrcnn_infer** function in **mrcnn_infer.py**.

3. mrcnn/02_merge_cells:
This profiles **merge_cells** function used in **mrcnn_infer** function in **mrcnn_infer.py**.
Because python line_profiler does not work on class method, function implementation is newly added in **mrcnn_infer.py**.

4. mrcnn/03_run_inference:
This profiles **run_inference** function in **stitched_inference** function in **inference.py**.

### Watershed-FPN
1. watershed/00_watershed_infer
This profiles **watershed_infer** function used in **demo.py**.
2. watershed/01_model_prediction
3. watershed/02_unet_predict

## The meaning of parameters in **config.ini**
1. ITER: the number of iterations. For example, if **ITER==100**, 100 copied images are used as input of **demo.py**
2. IMG_PATH: the path for the input image. The current version only support the same image.
3. FILE_NAME: the target file we want to benchmark. i.e., the file which has a line we want to profile.
4. FUNCTION_API: the part of line we want to profile.
5. FUNCTION_NAME: the name of function we want to profile.
