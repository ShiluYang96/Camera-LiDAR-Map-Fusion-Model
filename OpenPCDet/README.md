# Late Fusion on Robotino & KITTI Dataset

This repo is an implementation of late fusion model (CLOCs) on Robotino Dataset.
For the late fusion on KITTI, please refer to [Late_Fusion_KITTI.md](docs/Late_Fusion_KITTI.md) .

## 0.    Preparation
#### 0.1  Install OpenPCDet v0.5.2

The code is partly based on [Open-PCdet](https://github.com/open-mmlab/OpenPCDet), 
please install based on [install.md](docs/INSTALL.md).
For detail introduction and use of OpenPCDet please refer to [origin_readme.md](docs/README.md) .

#### 0.2   LiDAR data preparation

a) Transfer raw LiDAR .txt file to .pcd format
```shell
(pcdet) ...:~/OpenPCDet/tools$ python ../data_process/process_lidar.py --data_path PATH/TO/LIDAR/DATA
```

b)  Annotation of LiDAR data

You can use 3D annotation tools, for example [SUSTechPOINTS](https://github.com/naurril/SUSTechPOINTS) to generate labels .json file for LiDAR data.


#### 0.3  Camera data preparation

a)   Annotation of image data

You can use 2D annotation tools, for example [labelImg](https://github.com/tzutalin/labelImg) to generate labels:

```shell
# Run labelImg tool
(pcdet) ...:~/OpenPCDet/tools$ labelImg

# Below "Save" button in the left toolbar, click "PascalVOC" button to switch to CreateML format. 
# (Yolo format has problem with multi image saving)

# Click 'Change Save Dir' in Menu/File, choose `dataset_dir/labels/train/`

# Click 'Open Dir' in Menu/File, choose image dir `dataset_dir/images/train/`
```

b)   Transfer of annotation format

At first, please arrange data in following structure:
```
...
└── image data
    ├── images
    │    ├── train        
    │    └── val                    
    ├── annos
    │    ├── train               
    │    └── val
```

Trasfer annotation to YOLO format (for YOLO network):
```shell
# configure the Class_name
# generate yolo_train.txt
(pcdet) ...:~/OpenPCDet/tools$ python ../data_process/image_anno_2_yolo.py --dataset_dir PATH/TO/DATASET/ROOT --target_dir train
# generate yolo_val.txt
(pcdet) ...:~/OpenPCDet/tools$ python ../data_process/image_anno_2_yolo.py --dataset_dir PATH/TO/DATASET/ROOT --target_dir val
```
Trasfer annotation to COCO format (for [mmdetection](https://github.com/open-mmlab/mmdetection)):

Configure the path in `data_process/yolo_to_coco.py`, and then run the script.


## 1.    Train of 2D Detector
You can train with any 2D detector. After training, please generate the 2D proposals (for both train and val sets) in the following format:
```
# file_name: frame_id.txt (e.g: 20220530-100817.txt)
Car -1 -1 -10 1133.50 278.19 1225.04 329.51 -1 -1 -1 -1000 -1000 -1000 -10 0.0150 
# Class -1 -1 -10 x_min y_min x_max y_max -1 -1 -1 -1000 -1000 -1000 confidence_score
```
If you use [mmdetection](https://github.com/open-mmlab/mmdetection) as 2D detector plattform, you can use `data_process/generate_2D_bbox.py` to generate 2D proposals.<br>
It is reconmanded to deactivate the NMS process and set score_threshold to 0.001, to get more proposals.

## 2.    Train of 3D Detector
We have implemented the Robotino dataset in OpenPCDet dataset format. Please follow the following steps:

#### 2.1 Arrange the data files in following structre
```
OpenPCDet
└── data
    ├── robotino
    │   ├── training                      
    │   │   ├── points                    # LiDAR point clouds (.pcd)
    │   │   ├── images                    # images, only for visualization (.png)
    │   │   └── labels                    # generated labels from 0.2.b (.json)
    │   ├── testing                      
    │   │   ├── points
    │   │   ├── images
    │   │   └── labels
```

#### 2.2 Train with OpenPCDet
```shell
# check the network config file in ./tools/cfgs/robotino_models
# train pointpillar net with pretrained weights
(pcdet) ...:~/OpenPCDet/tools$ python train.py --cfg_file ./cfgs/robotino_models/pointpillar.yaml --batch_size 1 --workers 1 --epochs 100 --pretrained_model ../checkpoints/pointpillar_7728.pth
# for the use of other network, please change the config file according to pointpillar.py
```

#### 2.3 Quick demo of the trained model
```shell
(pcdet) ...:~/OpenPCDet/tools$ python demo.py --cfg_file ./cfgs/robotino_models/pointpillar.yaml --ckpt ./output/pointpillar/default/ckpt/checkpoint_epoch_100.pth --data_path ../data/robotino/training/points
```

#### 2.4 Generate 3D proposals

Please deactivate the NMS process, or set the NMS & score threshold in config file to very small.<br>
Generate 3D proposals for training & testing sets. (You can put train & val data in one folder, or change the data path in config file)
```shell
(pcdet) ...:~/OpenPCDet/tools$ python test.py --cfg_file ./cfgs/robotino_models/pointpillar.yaml --batch_size 1 --ckpt ./output/pointpillar/default/ckpt/checkpoint_epoch_100.pth --generate_output True
# eval 
(pcdet) ...:~/OpenPCDet/tools$ python test.py --cfg_file ./cfgs/robotino_models/pointpillar.yaml --batch_size 1 --ckpt ./output/pointpillar/default/ckpt/checkpoint_epoch_100.pth
```


## 3.    Train of Late Fusion Model

#### 3.1  Prepare 2D and 3D Proposals

Please arrange the data in the specific folder:
```
OpenPCDet
└── data
    ├── 2D_proposals                # 2D proposals (.txt)
    ├── 3D_proposals                # 3D proposals (.pt)
 ......
```

#### 3.2  Train fusion network

Modify the cfg, paths in `late_fusion/robotino_train.py` and run:
```shell
# if first time run, set generate to True to generate input_tensor
(pcdet) ...:~/OpenPCDet/tools$ python ../late_fusion/robotino_train.py  --generate True
```

All generated input data will saved in `data/input_tensor`, all log, eval information and checkpoints will be saved in `log`.

#### 3.3    Evaluation

Modify and run file `late_fusion/robotino_eval.py`, it will validate all checkpoints, you can choose the best one.

#### 3.4 Visualization

Modify the data_path and result_file in `late_fusion/visualization.py`, and then run. It will show the 2D bounding box on image and 3D bounding on lidar point clouds.

## 4. Commmon Problems

**a) FileNotFoundError: [Errno 2] No such file or directory: 'cfgs/dataset_configs/robotino_dataset.yaml'<br>**
**or: ModuleNotFoundError: No module named 'late_fusion'**
```shell
export PYTHONPATH=$PYTHONPATH:'/dir/to/your/OpenPCDet/'
export PYTHONPATH=$PYTHONPATH:'/dir/to/your/OpenPCDet/tools/'
```
## 5. Acknowledgement

[CLOCs_LQS](https://github.com/Laiqingsi/CLOCs_LQS) <br>
[CLOCs](https://github.com/pangsu0613/CLOCs) <br>
[OpenPCDet](https://github.com/open-mmlab/OpenPCDet)



