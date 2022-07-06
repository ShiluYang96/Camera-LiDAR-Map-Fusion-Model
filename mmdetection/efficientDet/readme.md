# EfficientDet Pytorch

```
├── README.md
├── train.py                               train skript
├── predict.py                             predict skript
├── eval.py                                evaluate skript
├── img                                    output of detection results during training
├── logs                            
├── utils                      
├── efficientdet
├── efficientnet
├── weights
├── others
│   ├── cal_mean_std.py                     hyper param tools
│   ├── kmeans_for_anchors.py
│   ├── kmeans_anchors_ratios.py
│   └── YOLO_to_COCO_format_transfer.py     annotation format transfer
├── config
│   └── door_handle_det.yaml                configutation file                
```
## 0.    Preparation
**0.1  Environment**
```
$ cd others
/others$ conda env create -f environment.yml
/others$ conda activate efficientDet
(efficientDet) .../others$ pip install -r requirements.txt
```

**0.2  Weights download**

Download pretrained weights in `weights` folder. The speed / FPS test includes the time of post-processing with no jit/data precision trick.

| coefficient | pth_download | GPU Mem(MB) | FPS | Extreme FPS (Batchsize 32) | mAP 0.5:0.95(this repo) | mAP 0.5:0.95(official) |
| :-----: | :-----: | :------: | :------: | :------: | :-----: | :-----: |
| D0 | [efficientdet-d0.pth](https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d0.pth) | 1049 | 36.20 | 163.14 | 33.1 | 33.8
| D1 | [efficientdet-d1.pth](https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d1.pth) | 1159 | 29.69 | 63.08 | 38.8 | 39.6
| D2 | [efficientdet-d2.pth](https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d2.pth) | 1321 | 26.50 | 40.99 | 42.1 | 43.0
| D3 | [efficientdet-d3.pth](https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d3.pth) | 1647 | 22.73 | - | 45.6 | 45.8
| D4 | [efficientdet-d4.pth](https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d4.pth) | 1903 | 14.75 | - | 48.8 | 49.4
| D5 | [efficientdet-d5.pth](https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d5.pth) | 2255 | 7.11 | - | 50.2 | 50.7
| D6 | [efficientdet-d6.pth](https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.0/efficientdet-d6.pth) | 2985 | 5.30 | - | 50.7 | 51.7
| D7 | [efficientdet-d7.pth](https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.2/efficientdet-d7.pth) | 3819 | 3.73 | - | 52.7 | 53.7
| D7X | [efficientdet-d8.pth](https://github.com/zylo117/Yet-Another-Efficient-Pytorch/releases/download/1.2/efficientdet-d8.pth) | 3983 | 2.39 | - | 53.9 | 55.1


## 1.    Train
**1.1  Prepare dataset**
```
    # your dataset structure should be like this
    datasets/
        -your_project_name/
            -train_set_name/
                -*.jpg
            -val_set_name/
                -*.jpg
            -annotations
                -instances_{train_set_name}.json
                -instances_{val_set_name}.json
    
    # for example, door_handle_detection
    Doors/
        -images/
            -train/
                -000000000001.jpg
                -000000000002.jpg
            -val/
                -000000000004.jpg
                -000000000005.jpg
            -annotations
                -instances_train.json (generated from 1.2)
                -instances_val.json   (generated from 1.2)
```

**1.2  Generate annotation**

Here we need to transfer the yolo-format-annotation we used before in COCO format.<br>
Configure the paths in `others/YOLO_to_COCO_format_transfer.py` and run.
```
   path_labels_train = os.fsencode("/path/to/yolo/annotation/train.txt")
   path_labels_val = os.fsencode("/path/to/yolo/annotation/val.txt")
   
   coco_format_save_path_train = '/path/to/output/annotation/instances_train.json'
   coco_format_save_path_val = '/path/to/output/annotation/instances_val.json'
   
   # Category file, one category per line
   yolo_format_classes_path = '/path/to/yolo/annotation/class.txt'
```
**1.3  Start training<br>**

Configure train parameters in `config/door_handle_det.yaml` file.<br>

Configure the `dataset_path` and `project` in `train.py`.<br>

Run `train.py`, checkpoints will be saved in `logs/project_name`, check loss information with tensorboard. <br>

1.3.1   Train a custom dataset from scratch --> w = None, head_only = False, debug = False
```
# train efficientdet-d0 on door_handle dataset 
(efficientDet) ...$ python train.py
```

1.3.2   Train a custom dataset with pretrained weights --> debug = False
```
# train efficientdet-d0 on door_handle dataset with pretrained weights
(efficientDet) ...$ python train.py -w /path/to/your/weights/efficientdet-d0.pth

# with a coco-pretrained, you can even freeze the backbone and train heads only
(efficientDet) ...$ python train.py -w /path/to/your/weights/efficientdet-d0.pth --head_only True
```

1.3.3    Resume training
```
# let says you started a training session like this.
(efficientDet) ...$ python train.py

# then you stopped it with a Ctrl+c, it exited with a checkpoint
# now you want to resume training from the last checkpoint
# simply set load_weights to 'last'
(efficientDet) ...$ python train.py -w last
```

1.3.4    Debug training (optional)
```
# when you get bad result, you need to debug the training result.
(efficientDet) ...$ python train.py --debug True

# then checkout img/ folder, there you can visualize the predicted boxes during training
# don't panic if you see countless of error boxes, it happens when the training is at early stage.
# But if you still can't see a normal box after several epoches, not even one in all image,
# then it's possible that either the anchors config is inappropriate or the ground truth is corrupted.
```

## 2.    Predict
**2.1 Configure the predict parameters in `config/door_handle_det.yaml`**

**2.2 Configure the `data_path`, choose prediction mode, run `predict.py`**
```
# predict single image
(efficientDet) ...$ python predict.py --mode 0 -w path/to/checkpoint.pth
Input image filename:img/street.jpg

# predict images in img_dir, the output images will be renamed and saved in img/
(efficientDet) ...$ python predict.py --mode 1 -w path/to/checkpoint.pth --img_dir path/to/val/images
```

## 3.    Evaluate
**3.1 Configure the evaluate parameters in `config/door_handle_det.yaml`**

**3.2 Configure `project`, `data_path` and run `eval.py`, the map output will be saved in `log/`**
```
(efficientDet) ...$ python eval.py -w /path/to/your/weights
```

## 4.    Hyper parameter setting
**4.1 Generate anchor scales and anchor ratio**

Cofigure the parameters in `config/door_handle_det.yaml`
```
img_dir = "path/to/images/train"
annotation_file = 'path/to/train.txt'
input_shape = [640, 640]
anchors_num = 3

computed paras:  ([0.234375, 0.5208333333333326, 1.195679453836151], [(1, 1.3665480427046266), (1, 1.0500000000000016), (1, 0.5063804919098074)])

```
The output computed paras: [anchor scales] and [anchor ratios]

**4.2 Fine tune the anchor ratios (optional)**
```
python others/kmeans_anchors_ratios.py \
--instances /home/yang/centerpoint_maps/2D_label_parser/instances_train.json \
--anchors-sizes 34 37 67 79 180 \
--input-size 768 \
--normalizes-bboxes True \
--num-runs 3 \
--num-anchors-ratios 3 \
--max-iter 300 \
--min-size 0 \
--iou-threshold 0.5 \
--decimals 1 \
--default-anchors-ratios '[(1, 0.9344729344729356), (1, 0.8460508701472562), (1, 0.9787234042553195)]'
```
**4.3 Generate the std and mean in RGB order (optional)**

Set image_dir in `cal_mean_std.py` and then run.

## 5. TODO

- [ ] log mAP every val interval
- [ ] Bug: conflict between float16=true and coco_eval during training
- [ ] test performance
- [X] code structure adaption
- [X] function test

## 6.    Common Problem
**a.    RuntimeError: CUDA out of memory. Tried to allocate 52.00 MiB (GPU 0; 15.90 GiB total capacity; 14.85 GiB already allocated; 51.88 MiB free; 15.07 GiB reserved in total by PyTorch)**<br>

GPU memory too small --> choose smaller batch size or smaller input image size

**b.    RuntimeError: Input type (torch.cuda.HalfTensor) and weight type (torch.cuda.FloatTensor) should be the same**<br>

Bug: conflict between float16 = true and coco_eval during training. Set float16 = false during training

**c.    RuntimeError: The model does not provide any valid output, check model architecture and the data input**<br>

Dataset is too small, network can not get prediction output for eval. Set `train.py:284: epoch > a larger value`. Or use the full data set.

## 6.    Reference

- [Original respository: zylo117](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch)
- [Source code: google/automl](https://github.com/google/automl)
- [lukemelas/EfficientNet-PyTorch](https://github.com/lukemelas/EfficientNet-PyTorch)
- [signatrix/efficientdet](https://github.com/signatrix/efficientdet)
- [vacancy/Synchronized-BatchNorm-PyTorch](https://github.com/vacancy/Synchronized-BatchNorm-PyTorch)
- [mnslarcher/kmeans-anchors-ratios](https://github.com/mnslarcher/kmeans-anchors-ratios)
