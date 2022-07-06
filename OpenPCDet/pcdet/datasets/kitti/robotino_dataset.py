import copy
import os.path
import pickle
import open3d as o3d
import numpy as np
from skimage import io
from pathlib import Path
import json
import glob
import math

from . import kitti_utils
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import box_utils, calibration_kitti, common_utils, object3d_kitti
from ..dataset import DatasetTemplate


class RobotinoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
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

        self.root_split_path = self.root_path / ('training' if training else 'testing')
        points_file_list = glob.glob(str(self.root_split_path / 'points' / '*.pcd'))
        labels_file_list = glob.glob(str(self.root_split_path / 'labels' / '*.json'))

        points_file_list.sort()
        labels_file_list.sort()
        self.sample_file_list = points_file_list
        self.sample_label_file_list = labels_file_list

        self.sample_id_list = [Path(self.sample_file_list[index]).stem for index in range(len(self.sample_file_list))]
        self.kitti_infos = []
        if not training:
            kitti_infos_val = self.get_infos(num_workers=4, has_label=True, count_inside_pts=True)
            self.kitti_infos.extend(kitti_infos_val)

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        sample_idx = Path(self.sample_file_list[index]).stem  # 0000.txt -> 0000 sample id n：number of cloud points m：number of labels
        # points = np.loadtxt(self.sample_file_list[index], dtype=np.float32).reshape(-1, 3)  # load point cloud: n*3
        pcd = o3d.io.read_point_cloud(self.sample_file_list[index])
        points = np.asarray(pcd.points, dtype=np.float32)

        # points_label = np.loadtxt(self.samplelabel_file_list[index], dtype=np.float32).reshape(-1, 7)  # label m*7
        with open(self.sample_label_file_list[index], 'r') as f:
            data = json.load(f)
        points_label = np.empty(shape=(0,7), dtype=np.float32)
        for obj in data:
            if obj['obj_type'] == 'Robotino':
                x = obj['psr']['position']['x']
                y = obj['psr']['position']['y']
                z = obj['psr']['position']['z']
                dx = obj['psr']['scale']['x']
                dy = obj['psr']['scale']['y']
                dz = obj['psr']['scale']['z']
                yaw = obj['psr']['rotation']['z']
                points_label = np.vstack([x,y,z,dx,dy,dz,yaw]).reshape(-1,7)

        gt_names = np.array(['Robotino'] * points_label.shape[0])

        input_dict = {
            'points': points,
            'frame_id': sample_idx,
            'gt_names': gt_names,
            'gt_boxes': points_label
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict

    def project_2D(self, lidar_box):
        """
        Args:
            lidar_box: [x, y, z, l, h, w, r]
        Returns:
            bbox: [x1,y1,x2,y2]
        """
        bbox = np.empty(shape=[lidar_box.shape[0], 4], dtype=np.float32)
        corners3d = box_utils.boxes3d_to_corners3d_kitti_camera(lidar_box)
        maxInColumns = np.amax(corners3d, axis=1) # x, y, z
        minInColumns = np.amin(corners3d, axis=1)

        for idx in range(maxInColumns.shape[0]):
            angle_1 = min(36 - math.atan2(maxInColumns[idx,1], maxInColumns[idx,0])/math.pi*180, 72.0)
            angle_2 = min(36 - math.atan2(minInColumns[idx,1], minInColumns[idx,0])/math.pi*180, 72.0)
            x_min = 640.0 * (max(angle_1, 0.) / 72.0)
            x_max = 640.0 * (max(angle_2, 0.) / 72.0)
            l = lidar_box[idx, 3]
            h = lidar_box[idx, 4]
            h_pixel = h / l * (x_max - x_min)
            y_min = max(480.0 / 2 - h_pixel, 0.)
            y_max = min(480.0 / 2 + h_pixel, 480.)
            bbox[idx, :] = [x_min, y_min, x_max, y_max]

        return bbox


    @staticmethod
    def generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None):
        """
        Args:
            batch_dict:
                frame_id:
            pred_dicts: list of pred_dicts
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path:
        Returns:
        """

        def get_template_prediction(num_samples):
            ret_dict = {
                'name': np.zeros(num_samples), 'truncated': np.zeros(num_samples),
                'occluded': np.zeros(num_samples), 'alpha': np.zeros(num_samples),
                'bbox': np.zeros([num_samples, 4]), 'dimensions': np.zeros([num_samples, 3]),
                'location': np.zeros([num_samples, 3]), 'rotation_y': np.zeros(num_samples),
                'score': np.zeros(num_samples), 'boxes_lidar': np.zeros([num_samples, 7])
            }
            return ret_dict

        def project_2D(lidar_box):
            """
            Args:
                lidar_box: [x, y, z, l, h, w, r]
            Returns:
                bbox: [x1,y1,x2,y2]
            """
            bbox = np.empty(shape=[lidar_box.shape[0], 4], dtype=np.float32)
            corners3d = box_utils.boxes3d_to_corners3d_kitti_camera(lidar_box)
            maxInColumns = np.amax(corners3d, axis=1)  # x, y, z
            minInColumns = np.amin(corners3d, axis=1)

            for idx in range(maxInColumns.shape[0]):
                angle_1 = min(36 - math.atan2(maxInColumns[idx, 1], maxInColumns[idx, 0]) / math.pi * 180, 72.0)
                angle_2 = min(36 - math.atan2(minInColumns[idx, 1], minInColumns[idx, 0]) / math.pi * 180, 72.0)
                x_min = 640.0 * (max(angle_1, 0.) / 72.0)
                x_max = 640.0 * (max(angle_2, 0.) / 72.0)
                l = lidar_box[idx, 3]
                h = lidar_box[idx, 4]
                h_pixel = h / l * (x_max - x_min)
                y_min = max(480.0 / 2 - h_pixel, 0.)
                y_max = min(480.0 / 2 + h_pixel, 480.)
                bbox[idx, :] = [x_min, y_min, x_max, y_max]

            return bbox

        def generate_single_sample_dict(batch_index, box_dict):
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict

            pred_boxes_camera = pred_boxes
            pred_boxes_img = project_2D(pred_boxes_camera)

            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            pred_dict['alpha'] = -np.arctan2(-pred_boxes[:, 1], pred_boxes[:, 0]) + pred_boxes_camera[:, 6]
            pred_dict['bbox'] = pred_boxes_img
            pred_dict['dimensions'] = pred_boxes_camera[:, 3:6]
            pred_dict['location'] = pred_boxes_camera[:, 0:3]
            pred_dict['rotation_y'] = pred_boxes_camera[:, 6]
            pred_dict['score'] = pred_scores
            pred_dict['boxes_lidar'] = pred_boxes

            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            frame_id = batch_dict['frame_id'][index]

            single_pred_dict = generate_single_sample_dict(index, box_dict)
            single_pred_dict['frame_id'] = frame_id
            annos.append(single_pred_dict)

            if output_path is not None:
                cur_det_file = output_path / ('%s.txt' % frame_id)
                with open(cur_det_file, 'w') as f:
                    bbox = single_pred_dict['bbox']
                    loc = single_pred_dict['location']
                    dims = single_pred_dict['dimensions']  # lhw -> hwl

                    for idx in range(len(bbox)):
                        print('%s -1 -1 %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f'
                              % (single_pred_dict['name'][idx], single_pred_dict['alpha'][idx],
                                 bbox[idx][0], bbox[idx][1], bbox[idx][2], bbox[idx][3],
                                 dims[idx][1], dims[idx][2], dims[idx][0], loc[idx][0],
                                 loc[idx][1], loc[idx][2], single_pred_dict['rotation_y'][idx],
                                 single_pred_dict['score'][idx]), file=f)

        return annos


    def get_infos(self, num_workers=4, has_label=True, count_inside_pts=True, sample_id_list=None):
        import concurrent.futures as futures

        def process_single_scene(sample_idx):
            print('%s sample_idx: %s' % ('test', sample_idx))
            info = {}
            pc_info = {'num_features': 4, 'lidar_idx': sample_idx}
            info['point_cloud'] = pc_info

            image_info = {'image_idx': sample_idx, 'image_shape': np.array([640, 480], dtype=np.int32)}
            info['image'] = image_info

            if has_label:
                # obj_list = self.get_label(sample_idx)
                label_file = str(self.root_split_path) + '/labels/' + sample_idx + '.json'
                with open(label_file, 'r') as f:
                    labels = json.load(f)

                points_label = np.empty(shape=(0, 7), dtype=np.float32)
                anno_location = np.empty(shape=[0, 3], dtype=np.float32)
                anno_dimension = np.empty(shape=[0, 3], dtype=np.float32)
                anno_rotation = np.empty(shape=0, dtype=np.float32)
                anno_alpha = np.empty(shape=0, dtype=np.float32)
                for obj in labels:
                    if obj['obj_type'] == 'Robotino':
                        x = obj['psr']['position']['x']
                        y = obj['psr']['position']['y']
                        z = obj['psr']['position']['z']

                        dx = obj['psr']['scale']['x']
                        dy = obj['psr']['scale']['y']
                        # l = max(dx, dy)
                        # w = min(dx, dy)
                        dz = obj['psr']['scale']['z']
                        yaw = obj['psr']['rotation']['z']
                        alpha = -math.atan2(-y, x) + yaw
                        anno_location = np.vstack([x,y,z]).reshape((-1,3))
                        anno_dimension = np.vstack([dx, dy, dz]).reshape((-1, 3))
                        anno_rotation = np.hstack([yaw])
                        anno_alpha = np.hstack([alpha])
                        points_label = np.vstack([x, y, z, dx, dy, dz, yaw]).reshape(-1, 7)

                annotations = {}
                annotations['name'] = np.array(['Robotino'] * anno_dimension.shape[0])
                annotations['truncated'] = np.zeros(shape=(anno_dimension.shape[0]), dtype=np.float32)
                annotations['occluded'] = np.zeros(shape=(anno_dimension.shape[0]), dtype=np.int32)
                annotations['alpha'] = anno_alpha
                annotations['bbox'] = self.project_2D(points_label)
                annotations['dimensions'] = anno_dimension  # lhw(camera) format
                annotations['location'] = anno_location
                annotations['rotation_y'] = anno_rotation
                annotations['score'] = np.zeros(shape=(anno_dimension.shape[0]), dtype=np.float32)
                annotations['difficulty'] = np.zeros(shape=(anno_dimension.shape[0]), dtype=np.int32)

                num_objects = len(labels)
                num_gt = len(annotations['name'])
                index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
                annotations['index'] = np.array(index, dtype=np.int32)
                annotations['gt_boxes_lidar'] = points_label

                info['annos'] = annotations

                if count_inside_pts:
                    """points = self.get_lidar(sample_idx)
                    calib = self.get_calib(sample_idx)
                    pts_rect = calib.lidar_to_rect(points[:, 0:3])

                    fov_flag = self.get_fov_flag(pts_rect, info['image']['image_shape'], calib)
                    pts_fov = points[fov_flag]
                    corners_lidar = box_utils.boxes_to_corners_3d(gt_boxes_lidar)
                    num_points_in_gt = -np.ones(num_gt, dtype=np.int32)

                    for k in range(num_objects):
                        flag = box_utils.in_hull(pts_fov[:, 0:3], corners_lidar[k])
                        num_points_in_gt[k] = flag.sum()"""
                    annotations['num_points_in_gt'] = np.ones(shape=(anno_dimension.shape[0]), dtype=np.int32)

            return info

        sample_id_list = sample_id_list if sample_id_list is not None else self.sample_id_list
        with futures.ThreadPoolExecutor(num_workers) as executor:
            infos = executor.map(process_single_scene, sample_id_list)
        return list(infos)

    def evaluation(self, det_annos, class_names, **kwargs):
        if 'annos' not in self.kitti_infos[0].keys():
            return None, {}

        from .kitti_object_eval_python import robotino_eval

        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.kitti_infos]
        ap_result_str, ap_dict = robotino_eval.get_official_eval_result(eval_gt_annos, eval_det_annos, class_names)

        return ap_result_str, ap_dict