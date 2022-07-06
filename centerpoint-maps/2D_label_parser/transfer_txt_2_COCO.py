"""
transfer generated 2D txt from nuScenes to COCO format
"""

import os
import time

from PIL import Image
from nuscenes.utils import splits
from tqdm import tqdm

from det3d.datasets.nuscenes.nusc_common_map import _get_available_scenes
import json
from nuscenes import NuScenes

root_path = "../data/nuScenes"
version = "v1.0-trainval"

coco_format_save_path_train = 'instances_train.json'
coco_format_save_path_val = 'instances_val.json'
label_root = "/home/yang/centerpoint_maps/2D_label_parser/target_labels"


def process(nusc, train_scenes, val_scenes, train_data=True):
    # filter exist scenes. you may only download part of dataset.
    available_scenes = _get_available_scenes(nusc)
    available_scene_names = [s["name"] for s in available_scenes]
    train_scenes = list(filter(lambda x: x in available_scene_names, train_scenes))
    val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes))
    train_scenes = set(
        [
            available_scenes[available_scene_names.index(s)]["token"]
            for s in train_scenes
        ]
    )
    val_scenes = set(
        [available_scenes[available_scene_names.index(s)]["token"] for s in val_scenes]
    )
    print("train scenes: ", len(train_scenes), "\nval scenes: ", len(val_scenes))

    # prepare info for coco json
    categories = [{'id': 1, 'name': "car", 'supercategory': "car"}]
    write_json_context = dict()
    write_json_context['info'] = {'description': '', 'url': '', 'version': '', 'year': 2022, 'contributor': '',
                                  'date_created': '2022-05-04 11:00:08.5'}
    write_json_context['licenses'] = [{'id': 1, 'name': None, 'url': None}]
    write_json_context['categories'] = categories
    write_json_context['images'] = []
    write_json_context['annotations'] = []

    file_number = 1
    num_bboxes = 1

    if train_data:  # generate for train data for val data
        target_scenes = train_scenes
        coco_format_save_path = coco_format_save_path_train
    else:
        target_scenes = val_scenes
        coco_format_save_path = coco_format_save_path_val

    for sample in tqdm(nusc.sample):
        channel_list = ['CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT', 'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT']

        if sample["scene_token"] in target_scenes:
            for ref_chan in channel_list:  # loop for each channel
                ref_sd_token = sample["data"][ref_chan]
                ref_cam_path, _, _ = nusc.get_sample_data(ref_sd_token)
                # get img size
                img_name, cam_dir = ref_cam_path.split('/')[-1],  ref_cam_path.split('/')[-2]
                txt_name = os.path.splitext(img_name)[0] + ".txt"
                label_path = os.path.join(label_root, cam_dir + '/' + txt_name)
                # get anno data

                img_context = {}
                im = Image.open(ref_cam_path)
                width, height = im.size
                # height, width = cv2.imread(img_path).shape[:2]
                img_context['file_name'] = cam_dir + '/' + img_name
                img_context['height'] = height
                img_context['width'] = width
                img_context['date_captured'] = '2022-05-04 11:00:08.5'
                img_context['id'] = file_number  # image id
                img_context['license'] = 1
                img_context['coco_url'] = ''
                img_context['flickr_url'] = ''
                write_json_context['images'].append(img_context)

                try:
                    with open(label_path, 'r') as f:
                        lines = f.readlines()

                    for i, line in enumerate(lines):  # for loop runs for number of annotations labelled in an image
                        bbox = line.split(' ')
                        bbox_dict = {}
                        class_id, x_center, y_center, w, h = bbox[0:5]
                        class_id, x_center, y_center, w, h = int(class_id), float(x_center), float(y_center), float(w), float(h)
                        x_center = x_center * width
                        w = w * width
                        y_center = y_center * height
                        h = h * height
                        x = x_center - w/2  # top left point of anchor
                        y = y_center - h/2

                        bbox_dict['id'] = num_bboxes
                        bbox_dict['image_id'] = file_number
                        bbox_dict['category_id'] = class_id + 1
                        bbox_dict['iscrowd'] = 0  # There is an explanation before
                        bbox_dict['area'] = h * w
                        x_coco = round(x)
                        y_coco = round(y)
                        if x_coco < 0:  # check if x_coco extends out of the image boundaries
                            x_coco = 1
                        if y_coco < 0:  # check if y_coco extends out of the image boundaries
                            y_coco = 1
                        bbox_dict['bbox'] = [x_coco, y_coco, w, h]
                        bbox_dict['segmentation'] = [
                            [x_coco, y_coco, x_coco + w, y_coco, x_coco + w, y_coco + h, x_coco, y_coco + h]]
                        write_json_context['annotations'].append(bbox_dict)
                        num_bboxes += 1
                except FileNotFoundError:  # if no labels
                    pass
                file_number = file_number + 1

    with open(coco_format_save_path, 'w') as fw:
        json.dump(write_json_context, fw)


if __name__ == "__main__":
    nusc = NuScenes(version=version, dataroot=root_path, verbose=True)
    train_scenes = splits.train
    val_scenes = splits.val

    print("###### start to process train labels ######")
    process(nusc, train_scenes, val_scenes)
    print("###### train labels completed! ######")
    time.sleep(1)
    print("###### start to process val labels ######")
    process(nusc, train_scenes, val_scenes, train_data=False)
    print("###### val labels completed! ######")
