from det3d.models.detectors.base import BaseDetector
from .base import BaseDetector
from ..registry import DETECTORS
from .. import builder
import torch 
from det3d.torchie.trainer import load_checkpoint


@DETECTORS.register_module
class MapPP(BaseDetector):
    def __init__(
        self,
        reader,
        backbone,
        neck,
        bbox_head,
        map_net, # new arguments 
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
    ):
        super(MapPP, self).__init__()
        self.reader = builder.build_reader(reader)
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)

        # now we load the model without the bbox head.  
        self.init_weights(pretrained=pretrained) # pretrained is the path to pretrained model 
        if pretrained is not None: # freeze weights only when pretrained checkpoint is provided
            self.freeze() # freeze everything except map_net and head 

        self.bbox_head = builder.build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.map_net = builder.build_map_net(map_net) # add an entry in the builder - VERIFY?

    def init_weights(self, pretrained=None):
        if pretrained is None:
            return 
        load_checkpoint(self, pretrained, strict=False) # we don't have bbox_head now 
    
        print("init weight from {}".format(pretrained))

    def extract_feat(self, data):
        input_features = self.reader(
            data["features"], data["num_voxels"], data["coors"]
        )
        x = self.backbone(
            input_features, data["coors"], data["batch_size"], data["input_shape"]
        )
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward(self, example, return_loss=True, **kwargs):
        voxels = example["voxels"]
        coordinates = example["coordinates"]
        num_points_in_voxel = example["num_points"]
        num_voxels = example["num_voxels"]

        batch_size = len(num_voxels)

        data = dict(
            features=voxels,
            num_voxels=num_points_in_voxel,
            coors=coordinates,
            batch_size=batch_size,
            input_shape=example["shape"][0],
        )
        mapdata = example['rasterized_map']  # img tensor (4, 1024, 1024, 3)
        x = self.extract_feat(data)

        # here  you get the map feature (the same shape as rpn output)
        map_feat = self.map_net(mapdata)  
        # print("x.shape = ", x.shape)  # [2, 384, 128, 128])
        # print("map_feat.shape = ", map_feat.shape)  # ([256, 3, 128, 128])
        x = torch.cat([x, map_feat], dim=1)  # tianwei had used dim = 1;PP raw map cat dims = [4, 384, 128, 128],  map fusion [2, 448, 128, 128]

        preds = self.bbox_head(x)

        if return_loss:
            return self.bbox_head.loss(example, preds)
        else:
            return self.bbox_head.predict(example, preds, self.test_cfg)

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False

        # FrozenBatchNorm2d.convert_frozen_batchnorm(self)
        return self