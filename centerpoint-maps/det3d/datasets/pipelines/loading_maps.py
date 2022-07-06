import os.path as osp
import warnings
import numpy as np
from functools import reduce

import pycocotools.mask as maskUtils

from pathlib import Path
from copy import deepcopy
from det3d import torchie
from det3d.core import box_np_ops
import pickle 

from ..registry import PIPELINES

import cv2

# TODO: Check if we need to do something specific to register the module


def read_map_img(path):
    
    img = cv2.imread(path) # img is an ndarray in BGR
    if img is None:
        print(f"NONE Img!!! {path}")
        img = np.zeros((128,128,3))
    else:
        # print(f"Img shape: {img.shape}; Type: {type(img)}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert to ndarray RGB 
    img = cv2.resize(img, (128,128))  # resize to desired shape (note that for basic map net we resize to 128x128 here)

    return img 

def get_obj(path):
    with open(path, 'rb') as f:
            obj = pickle.load(f)
    return obj 

@PIPELINES.register_module
class LoadRasterizedMapFromFile(object):
    def __init__(self):
        pass 
    def __call__(self, res, info):
        
        if res["type"] == "NuScenesDataset": 
            map_path = Path(info["map_path"])
            res["rasterized_map"] = read_map_img(str(map_path))
        else:
            raise NotImplementedError

        return res, info


