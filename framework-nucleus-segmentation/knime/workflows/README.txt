
10/28/-2019:
George Zaki

- HiTIF_CV7000_Nucleus_Segmentation_DeepLearning_IncResV2FPN_watershed2_serial_npy.knwf
Same as watershed2_serial, but the network weights are saved as numpy arrays instead of h5 files.


08-02-2019:

Workflows description:

1- HiTIF_AugmentInputGT_H5_OutLoc_JSON.knwf
Image augmentation workflow that does scaling, ROI selection, imgaug augmentation, and h5 generation

2- HiTIF_Calculate_mAP_GTvsInference_Python_3Inputs_OutLoc_JSON.knwf
Mean Average Precision calculation workflow

3- HiTIF_CV7000_Nucleus_Segmentation_DeepLearning_IncResV2FPN_GBDMsWS_nonSLURM_37X_OutLoc_JSON.knwf
Watershed 2 Segmentation workflow (Reddy's version)

4- HiTIF_CV7000_Nucleus_Segmentation_DeepLearning_IncResV2FPN_watershed2_serial.knwf
Watershed 2 segmentation workflow that uses 1 GPU and does two inferences in serial mode
