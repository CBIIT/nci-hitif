
unet_config = os.path.join(configs_dir, "{output}-config.cfg") 
rule train_unet:
    input:
        h5 = combined_h5,
        cfg = unet_config
    output:
        h5 = train_unet_h5,
        json = train_unet_json 
    params:
        training_dir = train_unet_dir
    run:
        train_unet_fpn(params.training_dir, input.cfg, input.h5, output.h5, output.json)

