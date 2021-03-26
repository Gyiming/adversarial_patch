# Adversarial Patch

This repo is for generating adversarial patch in continuous video sequence. 

## Dataset Preparation.
1. Download dataset from http://redwood-data.org/indoor_lidar_rgbd/download.html, RGB-D sequence.
2. Unzip the sequence into this folder. For example, if you are using the bedroom sequence, it should have two folder: image, depth under the ```./bedroom``` folder.
3. Create a new folder under ```./bedroom```, name it ```./bedroom/depth_ori```.
4. Create a new folder under ```./bedroom```, name it ```./bedroom/patch``` to store projected patch.
4. Copy everything in the ```./bedroom/depth``` folder into ```./bedroom/depth_ori``` for backup.


## Environment Setup.
1. ```conda env create -f env.yaml```
2. Install package Open3D from http://www.open3d.org/docs/release/getting_started.html, pip installation recommonded.

## Using Code.
1. Fill pixels without depth. ```python fill_depth.py --img xxx```. xxx is the image id.
2. Patch transformation. ```python test_depth.py --img1 xxx --img2 xxx --org True/False```. xxx is the image id. The flag org should True when you want to create a new patch in image 1 and transfer it to iamge 2, otherwise it means use pre-generated patch in image 1 and transfer it to image 2.