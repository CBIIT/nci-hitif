

#mrcnn training
train_mrcnn= "/data/HiTIF/data/dl_segmentation_paper/code/python/mask-rcnn/mask-rcnn-latest/train/train-wrapper.sh"
mrcnn_config = os.path.join(configs_dir, "mrcnn-config.cfg") 
mrcnn_train_log = os.path.join(train_mrcnn_dir, "mrcnn_train.log")



rule train_mrcnn:
    input:
        h5 = combined_h5,
        cfg = mrcnn_config
    output:
        model_h5 = train_mrcnn_model_h5_file
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



