# Inference of Cell Object Detection and Segmentation
This is an inference implementation on Python 3, Keras, and TensorFlow. The model generates bounding boxes and segmentation masks for each instance of an object in the image. It's based on Feature Pyramid Network (FPN), ResNet101, MRCNN, Gaussian Blurred border, and Distance Map model.

##Input:
![](assets/sample1.png)
##Output:
![](assets/sample2.png)

The folder includes:
* Jupyter notebooks of inference built on [MRCNN](https://arxiv.org/abs/1703.06870).
* Jupyter notebooks of inference built on Gaussian Blurred border and Distance Map Models.

# Installation
1. Clone this repository
```bash
   git@github.com:CBIIT/nci-hitif.git
   ```
2. Install dependencies
```bash
   pip3 install -r framework-nucleus-segmentation/mrcnn/requirements.txt
   ```
3. Download datasets.
```bash
   python3 framework-nucleus-segmentation/visualization/Download-and-Unzip.py
   ```

# Getting Started
* Jupyter notebook (MRCNN):[Here](https://github.com/CBIIT/nci-hitif/blob/master/framework-nucleus-segmentation/inference/mrcnn/demo/demo.ipynb)

* Jupyter notebook (Watershed):[Here](https://github.com/CBIIT/nci-hitif/blob/master/framework-nucleus-segmentation/inference/watershed/demo/demo.ipynb)
