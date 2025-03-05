# D2U-Net-Hybrid-U-Net
D2U-Net: A Dual Path Hybrid UNet Architecture for Precise Medical Image Segmentation.
This repository contains the implementation of the D2U-Net: A Dual Path Hybrid UNet Architecture, as described in our paper. The model is designed for high-performance medical image segmentation tasks, such as brain tumor, skin lesion, and polyp segmentation. Furthermore, this code can be applied to wide range of meical image segmentation datasets. 

Instructions: Clone the repository: git clone https://github.com/nooriahmed/D2U-Net-Hybrid-U-Net 

Key instructions:

If find the issues of overfitting during training then restart training with a small batch size.
Please follow the same pattern of folder settings of datasets as used in our code.
check layers shapes if resize images has been done.
Please note that some parts of the code are taken from given link, Please cite it as well when citing our work. #https://github.com/manideep2510/melanoma_segmentation
During training, please make sure to avoid any class imbalance inside the mentioned datasets, in such case loss function could be in negative during training.
In case of any uncertainity of classes or labelling in datasets or image color occlusions. A small difference in results may occur.
Follow the instructions in README.md to train and evaluate the model 

Requirements: 
Python vesion 3.10.12 
TensorFlow version 2.17.0
Keras 3.4.1
