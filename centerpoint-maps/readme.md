# LiDAR-Camera-Map Fusion Model Based on CenterPoint


This repo is modified from [centerpoint-maps](https://github.com/mon95/centerpoint-maps), which concatenated HD Map features with LiDAR data to enhance 3D object detection.<br>
We optimized the fusion process with more novel and light-weight MapFusionNet, and further promoted the accuracy with late fusion model CLOCs.

```
├── README.md
├── all_test.py                            eval skript for centerpoint
├── late_fusion_train.py                   train skript for late fusion
├── late_fusion_eval.py                    eval skript for late fusion
├── sample_token.py                        build subset dataset for nuScenes            
├── 2D_label_parser                        tool for transfer 3D anno to 2D                      
├── configs                                configuration files
├── data                                   dataset root path
├── late_fusion                            lib of late fusion model
├── det3d                                  lib of det3d
├── NuScenesVisualizeTool                  visualization tool for nuScnenes  
└── others                                 liscenes, env, requirements...
```

## 0.    Preparation
**0.1  Environment** 

```bash
$ cd others
/others$ conda env create -f environment.yml
/others$ conda activate centerpoint
(centerpoint) .../others$ pip install -r requirements.txt
```

**0.2  Advance Installation**

For advance installation, please refer to [INSTALL](others/INSTALL.md) to build `Cuda Extensions`.

**0.3  Data Preparation**

Download nuScenes dataset and organize as follows:

```   
data     
└── nuScenes
       ├── samples          <-- key frames
       ├── sweeps           <-- frames without annotation
       ├── maps             <-- map data & map extension
       ├── maps_generated   <-- empty
       ├── v1.0-trainval    <-- metadata
```

If you want to use subset of the nuScenes Dataset, please configure the path in `sample_token.py` and run script, to rebuild the annotation json and replace them in `data/nuScenes/v1.0-trainval`.

Create data and HD map:
``` bash
# create train-val data
$ python tools/create_data_map.py nuscenes_data_prep --root_path=data/nuScenes --version="v1.0-trainval" --nsweeps=10
# generate semantic HD maps
$ python tools/generate_HD_map.py create_HDmap --root_path=data/nuScenes --version="v1.0-trainval" --raw False
# if you want to generate raw HD maps
$ python tools/generate_HD_map.py create_HDmap --root_path=data/nuScenes --version="v1.0-trainval"

``` 
In the end, the data and info files should be organized as follows

```
└── CenterPoint
       └── data    
              └── nuScenes 
                     ├── samples          <-- key frames
                     ├── sweeps           <-- frames without annotation
                     ├── maps             <-- map data & map extension
                     ├── maps_generated   <-- generated map images
                     ├── v1.0-trainval    <-- metadata
                     |── infos_train_10sweeps_withvelo_filter_True.pkl <-- train annotations
                     |── infos_val_10sweeps_withvelo_filter_True.pkl <-- val annotations
                     |── dbinfos_train_10sweeps_withvelo.pkl <-- GT database info files
                     |── gt_database_10sweeps_withvelo <-- GT database 
```

The pretrained weigts are listed in [MODEL_ZOO.md](others/MODEL_ZOO.md)

## 1.    Train & Eval

At first, please edit the configuration file in `config`. 
```bash
# train with one gpu
$ python ./tools/train.py CONFIG_PATH
# distribute train with 2 gpu
$ python -m torch.distributed.launch --nproc_per_node=2 ./tools/train.py CONFIG_PATH

# eval single checkpoint
$ python ./tools/dist_test.py CONFIG_PATH --checkpoint CHECKPOINT_PATH --work_dir WORK_DIR --speed_test
# eval checkpoints in predifined dir, set config, work_dir and then run
$ python all_test.py
```
For tracking and test set please refer to [GETTING_START](others/GETTING_START.md). These are not tasks of our project.

## 2.    Visualization

For visualation, please run
```bash
$ python ./NuScenesVisualizeTool/visualize.py --result RESULT_JSON --root DATA_ROOT
```

## 3.    Late Fusion

**3.1 Train the 2D Detector on nuScenes**

Transfer nuScenes annotation to 2D:
```bash
$ python ./2D_label_parser/export_2D_anno_as_json.py --dataroot data/nuscenes --version v1.0-trainval --filename labels/2D-box.json
```

Transfer nuScenes 2D annotation to YOLO txt format:

Create the 6 camera channel dirs "CAM_BACK", "CAM_FRONT"... under `2D_label_parser/target_labels`. Then run
```bash
$ python ./2D_label_parser/label_parser.py -dt nuscenes -s ./target_labels/
```
Build COCO format annotation:

Configure the path and name in `2D_label_parser/transfer_txt_2_COCO.py`, the run the script.

Until now, you can get COCO format annotation for nuScenes, with that, you can use mmdetection to train and generate 2D proposals.

**3.2 Generate 3D Proposals**

Config the path in `all_test.py`, set `generate_3D_proposals` to True and then run script.

Tips: there should be only one valid checkpoint in work_dir.

**3.3 Train and Eval Late Fusion Model**

```bash
# generate input tensor and train
$ python late_fusion_train.py --d2path 2D_PROPOSAL --d3path 3D_PROPOSAL --generate True

# eval
$ python late_fusion_eval.py --checkpoint CHECKPOINT --d2path 2D_PROPOSAL --d3path 3D_PROPOSAL --generate True
```


## 4.    Reference

* [centerpoint_maps](https://github.com/mon95/centerpoint-maps)
* [CLOCs_LQS](https://github.com/Laiqingsi/CLOCs_LQS)
* [CLOCs](https://github.com/pangsu0613/CLOCs)
* [det3d](https://github.com/poodarchu/det3d)
* [second.pytorch](https://github.com/traveller59/second.pytorch)
* [CenterTrack](https://github.com/xingyizhou/CenterTrack)
* [CenterNet](https://github.com/xingyizhou/CenterNet) 
* [mmcv](https://github.com/open-mmlab/mmcv)
* [mmdetection](https://github.com/open-mmlab/mmdetection)
* [maskrcnn_benchmark](https://github.com/facebookresearch/maskrcnn-benchmark)
* [PCDet](https://github.com/sshaoshuai/PCDet)
* [OpenPCDet](https://github.com/open-mmlab/OpenPCDet)
