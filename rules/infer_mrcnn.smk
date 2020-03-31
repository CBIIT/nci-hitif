

#mrcnn_infernce

#My personal inference script
#infer_mrcnn="/gpfs/gsfs11/users/zakigf/mask-rcnn-with-augmented-images/Mask_RCNN/images/cell-images/inference/hitif_ml_segmentation/utils/run_hitif_inference.sh"

infer_mrcnn="/data/HiTIF/data/dl_segmentation_paper/code/python/mask-rcnn/mask-rcnn-latest/inference/run_hitif_inference.sh"

mrcnn_images_dir=os.path.join(inference_dir, "mrcnn","{exp}")

mrcnn_infer_log = os.path.join(mrcnn_images_dir, "mrcnn_inference-{exp}.log")

rule infer_mrcnn:
    input:
        #model_h5 = rules.train_mrcnn.output.model_h5, 
        model_h5 = mrcnn_model_h5_file,
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
        if not os.path.isdir(output.out_dir):
            print("Creating output directory")
            os.mkdir(output.out_dir)
        for image in input.images:
            print("Infer image:{0}".format(image))
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

