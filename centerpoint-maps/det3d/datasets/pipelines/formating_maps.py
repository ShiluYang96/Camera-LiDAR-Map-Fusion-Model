from det3d import torchie
import numpy as np
import torch

from ..registry import PIPELINES


class DataBundle(object):
    def __init__(self, data):
        self.data = data


@PIPELINES.register_module
class ReformatWithRasterizedMap(object):
    def __init__(self, **kwargs):
        double_flip = kwargs.get('double_flip', False)
        self.double_flip = double_flip 

    def __call__(self, res, info):
        meta = res["metadata"]
        points = res["lidar"]["points"]
        voxels = res["lidar"]["voxels"]
    
        rasterized_map = res["rasterized_map"]
    
        data_bundle = dict(
            metadata=meta,
            points=points,
            voxels=voxels["voxels"],
            shape=voxels["shape"],
            num_points=voxels["num_points"],
            num_voxels=voxels["num_voxels"],
            coordinates=voxels["coordinates"],
            rasterized_map = rasterized_map
        )

        if "anchors" in res["lidar"]["targets"]:
            anchors = res["lidar"]["targets"]["anchors"]
            data_bundle.update(dict(anchors=anchors))

        if res["mode"] == "val":
            data_bundle.update(dict(metadata=meta, ))

        calib = res.get("calib", None)
        if calib:
            data_bundle["calib"] = calib

        if res["mode"] != "test":
            annos = res["lidar"]["annotations"]
            data_bundle.update(annos=annos, )

        if res["mode"] == "train":
            # ground_plane = res["lidar"].get("ground_plane", None)
            #if ground_plane:
            #    data_bundle["ground_plane"] = ground_plane

            if "reg_targets" in res["lidar"]["targets"]: # anchor based
                labels = res["lidar"]["targets"]["labels"]
                reg_targets = res["lidar"]["targets"]["reg_targets"]
                reg_weights = res["lidar"]["targets"]["reg_weights"]

                data_bundle.update(
                    dict(labels=labels, reg_targets=reg_targets, reg_weights=reg_weights)
                )
            else: # anchor free
                data_bundle.update(res["lidar"]["targets"])

        elif self.double_flip:
            raise NotImplementedError # Intentionally removed this to make sure that we don't use this code path, i.e., we want to ignore double_flip

        return data_bundle, info
