# PyTorch_Projects

## Pretrained model repository for Heterogeneous Architecture Models on Pytorchwith CPU/CUDA

### Dependencies 

* Pytorch 1.0
* CUDA 9.0
* Python 2.7.12
* Numpy 1.15
* Torchvision 0.2.1
* Matplotlib 2.2.3

**NOTE : Jetson TX2 Development kit (Nvidia Tegra X2 GPU) with Jetpack 3.3 is used for these results**

Source code execution :
python Model.py --gpu 0/1 --file /Path/to/Image

For more paremeters type:
python Model.py --help

Example:
cd AlexNet
python AlexNet.py --gpu 1 --file /Path/to/Model/Folder 

### Timeline Logbook

* 10/01/2019 : Currently just partitioning between CPU/GPU per layer on LeNet5 and Fully CPU or fully GPU on AlexNet and ResNet(152)

* 23/01/2019 : Able to get video/images from terminal using CSI Camera. However, Gstreamer is not supported by current OpenCV L4T version on Jetson TX2. OpenCV must be recompiled supporting Gstreamer
