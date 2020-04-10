# Practical Deep Learning for Nucleus Segmentation

### Abstract:
<div style="text-align: justify">
Deep learning is rapidly becoming the technique of choice for automated segmentation of nuclei in biological image analysis workflows. In order to improve and understand the training parameters that drive the performance of deep learning models trained on small, custom annotated image datasets, we have designed a computational pipeline to systematically test different nuclear segmentation model architectures and model training strategies. Using this approach, we demonstrate that transfer learning and tuning of training parameters, such as the training image dataset composition, size and pre-processing, can lead to robust nuclear segmentation models, which match, and often exceed, the performance of existing, state-of-the-art deep learning models pre-trained on large image datasets. Our work provides computational tools and a practical framework for the improvement of deep learning-based biological image segmentation using small annotated image datasets. 
</div>

### Repo description:
The pipeline is shared as a reference implementation in this repo. 
The [inference](./inference), [visualization](./visualization), and [supervisely](./supervisely-wrapper) scripts can run on typical laptop.

Here is the description of the repo:

* [Inference](./inference): examples on how to use the models in inference mode.

* [Supervisely Wrapper](./supervisely-wrapper): code required for exporting/importing preminelary annotation and ground truth from supervisely.

* [Image Augmentation](image-augmentation): configurable augmentation wrapper implemented around imgaug.

* [MRCNN](./mrcnn): a forked version of Matterport MRCNN implemenation with our addition for the model's parameter we used for training, and the postprocessing pipeline after inference. The inference part is demonstrated in the [inference](./inference) directory.

* [Feature Pyramid Netwoks](./fpn): configurable wrappers, generators, and training around models we used to predict the distance transform and the blurred contour. Used used the [segmenation-models](https://github.com/qubvel/segmentation_models) library.

* [Pipeline](./pipeline): the Snakemake pipeline that we used to conduct our experiments on [NIH Biowulf](https://hpc.nih.gov/) HPC cluster. Some of the rules in the pipeline uses [Knime](https://www.knime.com/) workflows shown in this [directory](./knime) This implementation is provided for reference only. 



