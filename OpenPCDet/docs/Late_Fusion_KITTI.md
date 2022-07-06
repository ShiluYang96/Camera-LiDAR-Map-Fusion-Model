## Late fusion model on KITTI (CLOCs)

CLOCs is a novel Camera-LiDAR Object Candidates fusion network. It provides a low-complexity multi-modal fusion framework that improves the performance of single-modality detectors. CLOCs operates on the combined output candidates of any 3D and any 2D detector, and is trained to produce more accurate 3D and 2D detection results.
```
@article{pang2020clocs,
  title={CLOCs: Camera-LiDAR Object Candidates Fusion for 3D Object Detection},
  author={Pang, Su and Morris, Daniel and Radha, Hayder},
  booktitle={2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year={2020}
  organization={IEEE}
}
```
## 0.    Preparation
**0.1  Install PCdet v0.5.2**

The code is partly based on [Open-PCdet](https://github.com/open-mmlab/OpenPCDet), 
please install based on [install.md](https://github.com/open-mmlab/OpenPCDet/blob/master/docs/INSTALL.md).


## 1.    Train
**1.1  Prepare 2D and 3D proposals**

All data you need can be downloaded from  [here](https://drive.google.com/drive/folders/13h8452vcq0Wc-7p2BGMeJDpteHCDTdNs?usp=sharing).
After that, please arrange the data in the specific folder:
```
OpenPCDet
└── data
    ├── clocs_data
    │   ├── 3D                      # 3D proposals (.pt)
    │   ├── 2D                      # 2D proposals (.txt)
    │   ├── index                   # train/trainval/val.txt
    │   ├── info                    # kitti_infos_trainval/val.pkl
    │   └── input_data              # empty (to storage generated input data)
```
**1.2  Train fusion network**

Modify the cfg, paths in `fusion_train.py` and run:
```
$ python fusion_train.py --generate True
```
All generated input data will saved in `input_data`, all log, eval information and checkpoints will be saved in `log`.

## 2.    Evaluation

Modify and run file `fusion_eval.py`, it will validate all checkpoints, you can choose the best one.


## 3. Visualization

Modify the data_path and result_file in `visualization.py`, and then run. It will show the 2D bounding box on image and 3D bounding on lidar point cloud.


## 4. Acknowledgement

This code in based on implementation of [CLOCs_LQS](https://github.com/Laiqingsi/CLOCs_LQS). <br>
Other references: 
[CLOCs](https://github.com/pangsu0613/CLOCs) , [PCDet](https://github.com/open-mmlab/OpenPCDet) , [MMdetection](https://github.com/open-mmlab/mmdetection)



