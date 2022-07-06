# Author: Zylo117
# Modified by S.Y

import json
import os

import argparse
import torch
import yaml
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from efficientdet.backbone import EfficientDetBackbone
from efficientdet.utils import BBoxTransform, ClipBoxes
from utils.utils import preprocess, invert_affine, postprocess, boolean_string


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('-p', '--project', type=str, default='nuScenes',
                    help='project file that contains parameters')
    ap.add_argument('-w', '--weights', type=str, default="/home/yang/mmdetection/efficientDet/logs/nuScenes/efficientdet-d2_39_18277.pth",
                    help='path of weights file')
    ap.add_argument('--override', type=boolean_string, default=True,
                    help='override previous bbox results file if exists')
    ap.add_argument('--data_path', type=str, default='/home/yang/centerpoint_maps/2D_label_parser/',
                            help='the root folder of dataset')
    args = ap.parse_args()
    return args


class Params:  # load config file
    def __init__(self, project_file):
        self.params = yaml.safe_load(open(project_file).read())

    def __getattr__(self, item):
        return self.params.get(item, None)


def evaluate_coco(out_filepath, img_path, image_ids, coco, params, model):
    results = []
    input_sizes = eval(params.input_sizes)
    max_size = input_sizes[params.compound_coef]
    nms_threshold = float(params.nms_threshold)
    score_threshold = 0.05

    regressBoxes = BBoxTransform()
    clipBoxes = ClipBoxes()

    for image_id in tqdm(image_ids):
        image_info = coco.loadImgs(image_id)[0]
        image_path = os.path.join(img_path, image_info['file_name'])
        ori_imgs, framed_imgs, framed_metas = preprocess(image_path, max_size=max_size,
                                                         mean=params.mean, std=params.std)
        x = torch.from_numpy(framed_imgs[0])

        if bool(params.cuda):
            x = x.cuda(params.device)
            if bool(params.float16):
                x = x.half()
            else:
                x = x.float()
        else:
            x = x.float()

        x = x.unsqueeze(0).permute(0, 3, 1, 2)
        with torch.no_grad():
            features, regression, classification, anchors = model(x)

        preds = postprocess(x,
                            anchors, regression, classification,
                            regressBoxes, clipBoxes,
                            score_threshold, nms_threshold)

        if not preds:
            continue

        preds = invert_affine(framed_metas, preds)[0]

        scores = preds['scores']
        class_ids = preds['class_ids']
        rois = preds['rois']

        if rois.shape[0] > 0:
            # x1,y1,x2,y2 -> x1,y1,w,h
            rois[:, 2] -= rois[:, 0]
            rois[:, 3] -= rois[:, 1]

            bbox_score = scores

            for roi_id in range(rois.shape[0]):
                score = float(bbox_score[roi_id])
                label = int(class_ids[roi_id])
                box = rois[roi_id, :]

                image_result = {
                    'image_id': image_id,
                    'category_id': label + 1,
                    'score': float(score),
                    'bbox': box.tolist(),
                }

                results.append(image_result)

    if not len(results):
        raise Exception('the model does not provide any valid output, check model architecture and the data input')

    if os.path.exists(out_filepath):
        os.remove(out_filepath)
    json.dump(results, open(out_filepath, 'w'), indent=4)


def _eval(coco_gt, image_ids, out_filepath):
    # load results in COCO evaluation tool
    coco_pred = coco_gt.loadRes(out_filepath)

    # run COCO evaluation
    print('BBox')
    coco_eval = COCOeval(coco_gt, coco_pred, 'bbox')
    coco_eval.params.imgIds = image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


if __name__ == '__main__':
    # get args
    args = get_args()
    data_path = args.data_path
    override_prev_results = args.override
    project_name = args.project

    # load config file
    params = Params(f'config/{project_name}.yaml')
    compound_coef = params.compound_coef
    nms_threshold = float(params.nms_threshold)
    use_cuda = bool(params.cuda)
    gpu = params.device
    use_float16 = bool(params.float16)
    obj_list = params.obj_list


    weights_path = f'weights/efficientdet-d{compound_coef}.pth' if args.weights is None else args.weights

    print(f'running coco-style evaluation on project {project_name}, weights {weights_path}...')

    SET_NAME = params.val_set
    VAL_GT = f'{data_path}instances_{SET_NAME}.json'
    VAL_IMGS = f'/home/yang/centerpoint_maps/data/nuScenes/samples/'
    MAX_IMAGES = 10000
    coco_gt = COCO(VAL_GT)
    image_ids = coco_gt.getImgIds()[:MAX_IMAGES]

    saved_path = 'logs/' + f'/{project_name}/'
    os.makedirs(saved_path, exist_ok=True)
    out_filepath = saved_path + f'{SET_NAME}_bbox_results.json'
    
    if override_prev_results or not os.path.exists(out_filepath):
        model = EfficientDetBackbone(compound_coef=compound_coef, num_classes=len(obj_list),
                                     ratios=eval(params.anchors_ratios), scales=eval(params.anchors_scales))
        model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
        model.requires_grad_(False)
        model.eval()

        if use_cuda:
            model.cuda(gpu)

            if use_float16:
                model.half()

        evaluate_coco(out_filepath, VAL_IMGS, image_ids, coco_gt, params, model)

    _eval(coco_gt, image_ids, out_filepath)
