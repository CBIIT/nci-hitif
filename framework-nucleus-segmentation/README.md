# Practical Deep Learning for Nucleus Segmentation

### Abstract:
Deep learning is rapidly becoming the technique of choice for automated segmentation of nuclei in biological image analysis workflows. In order to improve and understand the training parameters that drive the performance of deep learning models trained on small, custom annotated image datasets, we have designed a computational pipeline to systematically test different nuclear segmentation model architectures and model training strategies. Using this approach, we demonstrate that transfer learning and tuning of training parameters, such as the training image dataset composition, size and pre-processing, can lead to robust nuclear segmentation models, which match, and often exceed, the performance of existing, state-of-the-art deep learning models pre-trained on large image datasets. Our work provides computational tools and a practical framework for the improvement of deep learning-based biological image segmentation using small annotated image datasets. 

### This repo:
Different parts of the pipeline are shared as a reference implementation in this github. 
The inference, visualization, and Supervisely directories can run on typical laptop.

Here is the description of the repo:

* [Supervisely Wrapper](./supervisely-wrapper): contains the code required for exporting/importing preminelary annotation and ground truth from supervisely.

* [mrcnn](./mrcnn): containes a forked version of Matterport with addition of the models we used for training, and the postprocessing done for inference. The inference part is demonstrated in the [inference](./inference) directory.

