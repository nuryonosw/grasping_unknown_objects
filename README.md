# Grasping Unknown Objects Using Convolutional Neural Networks.

This repository consists of two grasping neural networks implemented in two approaches and the experimental results. Each neural network is a separate package. The project focuses on improving two existing grasping neural networks to perform better and generic in the 3D world. Experiments were also performed to evaluate and compare the performance in various cases. All experiments were performed in a table top scene. This implementation is for the PAL robotics, Tiago robot. It can be used on other robots by changing the topic names and arm group name. 


## Getting started

Clone this repository into the catkin workspace.

```bash
git clone https://github.com/Pranav24-8/grasping_unknown_objects.git
```

### Generative Grasping CNN

This neural network is present in the package ```ggcnn_grasp```. The following steps can be used to get this running.

```bash
cd ws19_prasad_grasping_objects/ggcnn_grasp/ggcnn
pip install -r requirements.txt
cd ..
mkdir Dataset
```

Now download and extract the [Cornell grasping dataset](http://pr.cs.cornell.edu/grasping/rect_data/data.php) into the Dataset folder. Then run the following command to create depth images in the dataset from the point clouds.

```bash
cd ggcnn
python -m utils.dataset_processing.generate_cornell_depth <Path To Dataset>
```

To run the training use the following command:

```bash
python train_ggcnn.py --description training_example --network ggcnn --dataset cornell --dataset-path <Path To Dataset>
```

The training stores a file for each epoch in  [ggcnn_grasp/ggcnn/output/models/](https://github.com/Pranav24-8/grasping_unknown_objects/tree/master/ggcnn_grasp/ggcnn/output/models) and every file has the validation score appended. Copy the file with maximum score to [ggcnn_grasp/src/opmodels](https://github.com/Pranav24-8/grasping_unknown_objects/tree/master/ggcnn_grasp/src/opmodels). Replace the file name in line 12 of [ggcnn_predict.py](https://github.com/Pranav24-8/grasping_unknown_objects/blob/master/ggcnn_grasp/src/ggcnn_predict.py) with the new one.

Install all other dependencies listed below and now this network is ready to predict grasps. 

```ggcnn_predict.launch``` - Runs the network prediction file.

```ggcnn_app1.launch``` - Runs the Approach 1 (Distinguish grasps based on object dimensions) grasping pipeline.

```ggcnn_app2.launch``` - Runs the Approach 2 (Defining grasps based on surface normals) grasping pipeline.

```ggcnn_realr.launch``` - Runs the real robot grasping pipeline. Need to launch darknet_ros separately.

### RCNN Multi Grasp

This neural network is present in the package ```rcnn_grasp```. The following steps can be used to get this running.

```bash
cd rcnn_grasp
cd lib
make clean
make
cd ..
```
Install the [Python COCO API](https://github.com/cocodataset/cocoapi).

```bash
cd data
git clone https://github.com/pdollar/coco.git
cd coco/PythonAPI
make
cd ../../..
```

To train a model,


*  Download [Cornell Dataset](http://pr.cs.cornell.edu/grasping/rect_data/data.php)

*  Run dataPreprocessingTest_fasterrcnn_split.m (please modify paths according to your structure)

```bash
./experiments/scripts/train_faster_rcnn.sh 0 graspRGB res50
```

Or a pre-trained model can be downloaded [here](https://www.dropbox.com/s/ldapcpanzqdu7tc/models.zip?dl=0).

```bash
mkdir output \
cd output \
mkdir res50 \
cd res50 \
mkdir train \
cd train \
mkdir default \
cd default \
```

The models need to be placed in output/res50/train/default/.

A demo can be tested using the following command:

```./src/demo_graspRGB.py --net res50 --dataset grasp```

Now this network is ready to predict grasps.

Grasp prediction is a part of the grasping pipeline for this network.

```rcnn_app1.launch``` - Runs the Approach 1 (Distinguish grasps based on object dimensions) grasping pipeline.

```rcnn_app2.launch``` - Runs the Approach 2 (Defining grasps based on surface normals) grasping pipeline.

```rcnn_realr.launch``` - Runs the real robot grasping pipeline. Need to launch darknet_ros separately.

## External dependencies
[Python-pcl](https://github.com/strawlab/python-pcl) ```pip install python-pcl```

CV-Bridge ```apt-get install ros-kinetic-cv-bridge```

ros_numpy ```apt-get install ros-kinetic-ros-numpy```

[darknet_ros](https://github.com/leggedrobotics/darknet_ros) Follow steps in the darknet_ros repository.

## Build dependencies

[DockerFile](https://fbe-gitlab.hs-weingarten.de/stud-iki/thesis-master/ws19_prasad_grasping_objects/blob/dev/docker/DockerFile)

## Run dependencies
Object detection - ```roslaunch darknet_ros darknet_ros.launch```

## Package Information

This package works on all ROS versions until kinetic. The python-pcl bindings were not updated for melodic yet. When updated, this package will also work on ROS melodic.

## Authors
Pranav Krishna Prasad - @pk-183384


