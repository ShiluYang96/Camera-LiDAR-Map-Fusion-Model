"""
author: Maples
https://blog.csdn.net/qq_16137569/article/details/121066977
modified: S. Y
"""
import argparse
import copy
import os
import json
import mmcv
import numpy as np
from nuscenes.nuscenes import NuScenes
from helper import *


def parse_args():
    parser = argparse.ArgumentParser(description="visualization of the detection result")
    parser.add_argument(
        "--result",
        type=str,
        default="/home/foj-sy/centerpoint-maps-master/result/late_fusion.json",
        help="results json from eval process",
    )
    parser.add_argument("--score_thres", type=float, default=0.4,
                        help="threshold, predicted results with scores above that will be visualized",)
    parser.add_argument("--root", type=str, default="/home/foj-sy/centerpoint-maps-master/data/nuScenes", help="data root",)
    parser.add_argument("--version", type=str, default="v1.0-trainval", help="version of nuscenes dataset",)
    parser.add_argument("--proj_lidar", type=bool, default=False, help="whether project lidar points on image",)
    args = parser.parse_args()
    return args


def traverse_images(nusc, score_thres, model_output=None, proj_lidar=False, visible_level=2, save_dir=None):
    """
    model_output: results json from eval
    proj_lidar: whether project lidar points on image plane
    visible_level: visible level of lidar points
    save_dir: dir to save output images
    """
    if model_output is None:
        # when no results, traverse the entire dir
        for scene in nusc.scene:
            sample = None
            while True:
                if sample is None:
                    sample = nusc.get('sample', scene['first_sample_token'])

                visualize_one_sample(nusc,
                                     sample,
                                     proj_lidar=proj_lidar,
                                     visible_level=visible_level,
                                     save_dir=save_dir)

                if sample['next'] != '':
                    sample = nusc.get('sample', sample['next'])
                else:
                    break
    else:
        # if there is detection results, iterate based on results
        for sample_token in mmcv.track_iter_progress(model_output['results']):
            results = model_output['results'][sample_token]
            sample = nusc.get('sample', sample_token)

            # visualization
            visualize_one_sample(nusc,
                                 sample,
                                 results,
                                 proj_lidar=proj_lidar,
                                 visible_level=visible_level,
                                 score_thres=score_thres,
                                 save_dir=save_dir)


if __name__ == '__main__':
    args = parse_args()
    result_file = args.result
    version = args.version
    proj_lidar = args.proj_lidar
    score_thres = args.score_thres
    with open(result_file) as f:
        model_outputs = json.load(f)
    data_root = args.root
    nusc = NuScenes(version=version,
                    dataroot=data_root,
                    verbose=True)

    traverse_images(nusc, score_thres, model_output=model_outputs, proj_lidar=proj_lidar, visible_level=1, save_dir=None)
    # check_dataset_info(nusc)
