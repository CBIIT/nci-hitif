# Inference of Cell Object Detection and Segmentation
This is an inference implementation on Python 3, Keras, and TensorFlow. The model generates bounding boxes and segmentation masks for each instance of an object in the image. It's based on Feature Pyramid Network (FPN), ResNet101, MRCNN, Gaussian Blurred border, and Distance Map model.

## Input:
![](assets/sample1.png)
## Output:
![](assets/sample2.png)

The folder includes:
* Jupyter notebooks of inference built on [MRCNN](https://arxiv.org/abs/1703.06870).
* Jupyter notebooks of inference built on Gaussian Blurred border and Distance Map Models.

# Installation
## 1. Clone this repository
```bash
  git clone git@github.com:CBIIT/nci-hitif.git
  cd nci-hitif
   ```
## 2-a. Install dependencies (Option 1)
```bash
   sudo pip install -r framework-nucleus-segmentation/mrcnn/requirements.txt
   ```
## 2-b. Install dependencies using Conda (Option 2)
### Create Virtual Conda Enviroment using Anaconda Navigator, and install all dependencies in [Here](https://github.com/CBIIT/nci-hitif/blob/master/framework-nucleus-segmentation/mrcnn/requirements.txt)

### Activate Virtual Environment on Terminal.
```bash
   conda activate <your-environment-name>
   ```
## 3. Run Dependency (MRCNN Package) setup from the MRCNN directory:
```bash
  cd framework-nucleus-segmentation/mrcnn
  python3 setup.py install
  cd ../..
   ```
## 4. Download datasets. Downloaded dataset will be located at **framework-nucleus-segmentation/visualization**
```bash
   python3 framework-nucleus-segmentation/visualization/Download-and-Unzip.py
   ```
## 5. Install JupyterLab (Recommended: Install using Anaconda Navigator). Now, you are ready for running demos.

# Source Code:
## MRCNN Inference:
```bash
   nci-hitif/framework-nucleus-segmentation/inference/mrcnn/src/mrcnn_infer.py
   ```
## FPN-Watershed Inference:
```bash
   nci-hitif/framework-nucleus-segmentation/inference/watershed/src/watershed_infer.py
   ```


# Demo
* Jupyter notebook (MRCNN): [Here](https://github.com/CBIIT/nci-hitif/blob/master/framework-nucleus-segmentation/inference/mrcnn/demo/demo.ipynb)

* Jupyter notebook (FPN-Watershed): [Here](https://github.com/CBIIT/nci-hitif/blob/master/framework-nucleus-segmentation/inference/watershed/demo/demo.ipynb)
