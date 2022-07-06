# Author: Zylo117
# Modified by S.Y

import argparse
import os
import time
from pathlib import Path

import torch
import yaml
from torch.backends import cudnn
from tqdm import tqdm

from efficientdet.backbone import EfficientDetBackbone
import cv2
import numpy as np

from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess, STANDARD_COLORS, standard_to_bgr, get_index_label, plot_one_box

def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('-p', '--project', type=str, default='nuScenes',
                    help='project file that contains parameters')
    ap.add_argument("--mode", help="mode for prediction. 0: single picture, 1: folder", type=int, default=1)
    ap.add_argument("--img_dir", help="image dir to be detected", type=str, default="/home/yang/centerpoint_maps/data/nuScenes/samples")
    ap.add_argument('-w', '--weights', type=str, default="/home/yang/mmdetection/efficientDet/logs/nuScenes/efficientdet-d1_48_22400.pth",
                    help='path of weights file')
    args = ap.parse_args()
    return args


class Params:  # load config file
    def __init__(self, project_file):
        self.params = yaml.safe_load(open(project_file).read())

    def __getattr__(self, item):
        return self.params.get(item, None)


def generate_proposals(preds, img_path, sub_out_dir):

    # get image name
    img = img_path.split('/')[-1]
    frame_id = img.split('.')[0]
    out_file = os.path.join(sub_out_dir, (frame_id + '.txt'))


    bboxes = []
    scores = []
    objs = []
    for j in range(len(preds['rois'])):
        x1, y1, x2, y2 = preds['rois'][j]
        obj = obj_list[preds['class_ids'][j]]
        score = float(preds['scores'][j])
        bboxes.append([x1, y1, x2, y2])
        scores.append(score)
        objs.append(obj)

        context = []
        for i in range(len(objs)):
            if (objs[i] == 'car') & (len(context) < 200):  # save 200 highest proposals
                bbox = bboxes[i]
                if ((i + 1) < len(objs)) & (len(context) < 199):
                    write_line = "Car -1 -1 -10 {:.2f} {:.2f} {:.2f} {:.2f} -1 -1 -1 -1000 -1000 -1000 -10 {:.4f} \n". \
                        format(bbox[0], bbox[1], bbox[2], bbox[3], scores[i])
                else:  # if the last line
                    write_line = "Car -1 -1 -10 {:.2f} {:.2f} {:.2f} {:.2f} -1 -1 -1 -1000 -1000 -1000 -10 {:.4f} ". \
                        format(bbox[0], bbox[1], bbox[2], bbox[3], scores[i])

                context.append(write_line)
        # write all 2d proposals
        with open(out_file, 'w+') as f:
            for line in context:
                f.write(line)


if __name__ == '__main__':
    # get args
    args = get_args()
    project_name = args.project
    mode = args.mode
    img_dir = args.img_dir

    # load config file
    params = Params(f'config/{project_name}.yaml')
    compound_coef = params.compound_coef
    nms_threshold = float(params.nms_threshold)
    use_cuda = bool(params.cuda)
    gpu = params.device
    cudnn.fastest = True
    cudnn.benchmark = True
    use_float16 = bool(params.float16)
    extreme_fps_test = bool(params.extreme_fps_test)
    obj_list = params.obj_list
    force_input_size = params.force_input_size
    weights_path = f'weights/efficientdet-d{compound_coef}.pth' if args.weights is None else args.weights
    anchor_ratios = eval(params.anchors_ratios)
    anchor_scales = eval(params.anchors_scales)
    threshold = params.score_threshold
    iou_threshold = params.iou_threshold
    out_dir = "Output_2D"

    channel_list = ['CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT', 'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT']
    for ref_chan in channel_list:
        sub_img_dir = os.path.join(img_dir, ref_chan)
        sub_out_dir = os.path.join(out_dir, ref_chan)
        # create output dir
        Path(sub_out_dir).mkdir(parents=True, exist_ok=True)
        image_path = os.listdir(sub_img_dir)
        img_paths = [os.path.join(sub_img_dir, img_path) for img_path in image_path]
        color_list = standard_to_bgr(STANDARD_COLORS)

        # tf bilinear interpolation is different from any other's, just make do
        input_sizes = eval(params.input_sizes)
        input_size = input_sizes[compound_coef] if force_input_size is 0 else force_input_size

        model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list),
                                     ratios=anchor_ratios, scales=anchor_scales)
        model.load_state_dict(torch.load(weights_path, map_location='cpu'))
        model.requires_grad_(False)
        model.eval()

        if use_cuda:
            model = model.cuda()
        if use_float16:
            model = model.half()

        with tqdm(total=len(img_paths)) as pbar:
            for img_path in img_paths:
                ori_imgs, framed_imgs, framed_metas = preprocess(img_path, max_size=input_size)

                if use_cuda:
                    x = torch.stack([torch.from_numpy(fi).cuda() for fi in framed_imgs], 0)
                else:
                    x = torch.stack([torch.from_numpy(fi) for fi in framed_imgs], 0)

                x = x.to(torch.float32 if not use_float16 else torch.float16).permute(0, 3, 1, 2)

                with torch.no_grad():
                    features, regression, classification, anchors = model(x)

                regressBoxes = BBoxTransform()
                clipBoxes = ClipBoxes()

                out = postprocess(x,
                                  anchors, regression, classification,
                                  regressBoxes, clipBoxes,
                                  threshold, iou_threshold)

                out = invert_affine(framed_metas, out)

                if len(out[0]['rois']) == 0:
                    continue
                generate_proposals(out[0], img_path, sub_out_dir)
                pbar.update(1)
