import argparse
import glob
import time
from pathlib import Path

import cv2
import open3d
from tools.visual_utils import open3d_vis_utils as V

OPEN3D_FLAG = True

import numpy as np
import torch
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.utils import common_utils


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--dataset', type=str, default='robotino',
                        help='choose dataset type: KITTI or robotino')
    parser.add_argument('--cfg_file', type=str, default='../tools/cfgs/robotino_models/pointpillar.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='../data/robotino/testing',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--result_file', type=str, default="../log/late_fusion/det_result/0.pt",
                        help='specify the output detection result file')
    parser.add_argument('--score_threshold', type=float, default=0.11,
                        help='only the anchor with confidence score larger than the threshold will be visualized')
    parser.add_argument('--ext', type=str, default='.pcd', help='specify the extension of your point cloud data file')
    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    return args, cfg


# load detection result
def get_det(data, frame_id, threshold):
    for det in data:
        if det["frame_id"] == frame_id:
            pred_scores = det["score"]
            mask = pred_scores >= threshold
            pred_scores = pred_scores[mask]
            pred_boxes = det["boxes_lidar"][mask,:]
            pred_2D_box = det["bbox"][mask,:]
            pred_labels = []
            for i in range(len(det["name"][mask])):
                pred_labels.append(1)
            return pred_boxes, pred_scores, pred_labels, pred_2D_box


def show_2D_bbox(img_dir, frame_id, bboxs):
    img_full_name = img_dir + "/" + frame_id + ".png"

    # load the image and scale it
    image = cv2.imread(img_full_name)
    thick = 3
    color = (0, 0, 255)
    cv2.namedWindow("image window", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("image window", 1200, 1200)
    for bbox in bboxs:
        left, top, right, bottom = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        cv2.rectangle(image, (left, top), (right, bottom), color, thick)
    cv2.imshow('image window', image)
    # add wait key. window waits until user presses a key
    cv2.waitKey(0)
    # and finally destroy/close all open windows
    cv2.destroyAllWindows()


class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        elif self.ext == '.pcd':
            pcd = open3d.io.read_point_cloud(self.sample_file_list[index])
            points = np.asarray(pcd.points, dtype=np.float32)
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        frame_id = self.sample_file_list[index].split('/')[-1]
        frame_id = frame_id.split('.')[0]
        return data_dict, frame_id


def main():
    args, cfg = parse_config()
    if args.dataset == 'KITTI':
        lidar_data = args.data_path + "/velodyne"
        img_data = args.data_path + "/image_2"
    else:
        lidar_data = args.data_path + "/points"
        img_data = args.data_path + "/images"
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    demo_dataset = DemoDataset(
        dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
        root_path=Path(lidar_data), ext=args.ext, logger=logger
    )
    logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    # read detection result .pt
    det_data = torch.load(args.result_file)

    for idx, (data_dict, frame_id) in enumerate(demo_dataset):
        # frame_id = str(idx).zfill(6)

        logger.info(f'Visualized sample index: \t{idx + 1}')
        data_dict = demo_dataset.collate_batch([data_dict])

        try:  # if the frame id in val list
            pred_boxes, pred_scores, pred_labels, pred_2D_box = get_det(det_data, frame_id, threshold=args.score_threshold)

            show_2D_bbox(img_data, frame_id, pred_2D_box)
            time.sleep(1)
            V.draw_scenes(
                points=data_dict['points'][:, 1:], ref_boxes=pred_boxes,
                ref_scores=pred_scores, ref_labels=pred_labels)
        except TypeError:
            pass

    logger.info('Demo done.')


if __name__ == '__main__':
    main()
