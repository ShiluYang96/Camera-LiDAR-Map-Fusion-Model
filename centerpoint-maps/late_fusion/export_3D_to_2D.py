import torch
import numpy as np
from pyquaternion import Quaternion
import cv2
import os

LIDAR_MIN_X = -20
LIDAR_MAX_X = 20
LIDAR_MIN_Y = -35
LIDAR_MAX_Y = 35


def get_obj3d_from_annotation(ann, ego_data, calib_data):
    obj_ann = dict()

    # 2. 3D bbox
    # global frame
    center = np.array(ann.center)
    orientation = np.array(ann.orientation)
    # transfer from global frame to ego vehicle frame
    quaternion = Quaternion(ego_data['rotation']).inverse
    center -= np.array(ego_data['translation'])
    center = np.dot(quaternion.rotation_matrix, center)
    orientation = quaternion * orientation
    # from ego vehicle frame to sensor frame
    quaternion = Quaternion(calib_data['rotation']).inverse
    center -= np.array(calib_data['translation'])
    center = np.dot(quaternion.rotation_matrix, center)
    orientation = quaternion * orientation
    # generate 3D bbox according to center point and roration
    x, y, z = center
    w, l, h = ann.wlh
    x_corners = l / 2 * np.array([-1, 1, 1, -1, -1, 1, 1, -1])
    y_corners = w / 2 * np.array([1, 1, -1, -1, 1, 1, -1, -1])
    z_corners = h / 2 * np.array([-1, -1, -1, -1, 1, 1, 1, 1])
    # initial center (0, 0, 0)
    box3d = np.vstack((x_corners, y_corners, z_corners))
    # rotate the 3D bbox
    box3d = np.dot(orientation.rotation_matrix, box3d)
    # shift 3D bbox
    box3d[0, :] = box3d[0, :] + x
    box3d[1, :] = box3d[1, :] + y
    box3d[2, :] = box3d[2, :] + z

    obj_ann['data_type'] = 'results'
    obj_ann['type'] = 'car'
    obj_ann['box'] = box3d

    return obj_ann


def project_obj2image(obj3d_list, intrinsic):
    obj2d_list = list()

    trans_mat = np.eye(4)
    trans_mat[:3, :3] = np.array(intrinsic)

    for obj in obj3d_list:
        # step1: check the projected box in the image field
        in_front = obj['box'][2, :] > 0.1
        if all(in_front) is False:
            continue

        # step2: trasfer to pixel frame
        points = obj['box']
        points = np.concatenate((points, np.ones((1, points.shape[1]))), axis=0)
        points = np.dot(trans_mat, points)[:3, :]
        points /= points[2, :]

        obj2d = {'data_type': obj['data_type'], 'type': obj['type'], 'box': points}
        obj2d_list.append(obj2d)

    return obj2d_list


def plot_annotation_info(lidar_bev, camera_img, obj_list, show_2D=True):
    assert (lidar_bev is None and camera_img is not None) \
           or (lidar_bev is not None and camera_img is None)

    for obj in obj_list:
        obj_type = obj['type']
        box = obj['box']

        thickness = [2, 3]
        if obj_type == 'car':
            color = (0, 255, 255)
        elif obj_type in ['truck', 'trailer', 'bus', 'construction_vehicle']:
            color = (64, 128, 255)
        elif obj_type == 'pedestrian':
            color = (0, 255, 0)
        elif obj_type in ['bicycle', 'motorcycle']:
            color = (255, 255, 0)
        else:
            continue

        if obj['data_type'] == 'gt':
            color = (255, 255, 255)
            thickness = [1, 2]

        if lidar_bev is not None:
            # lidar bev
            img_h, img_w, _ = lidar_bev.shape
            for i in range(4):
                j = (i + 1) % 4
                u1 = int((box[0, i] - LIDAR_MIN_X) / (LIDAR_MAX_X - LIDAR_MIN_X) * img_w)
                v1 = int(img_h - (box[1, i] - LIDAR_MIN_Y) / (LIDAR_MAX_Y - LIDAR_MIN_Y) * img_h)
                u2 = int((box[0, j] - LIDAR_MIN_X) / (LIDAR_MAX_X - LIDAR_MIN_X) * img_w)
                v2 = int(img_h - (box[1, j] - LIDAR_MIN_Y) / (LIDAR_MAX_Y - LIDAR_MIN_Y) * img_h)
                cv2.line(lidar_bev, (u1, v1), (u2, v2), color, thickness=thickness[0])
        else:
            # camera views
            box = box.astype(np.int)
            if not show_2D:
                for i in range(4):
                    j = (i + 1) % 4
                    # underside
                    cv2.line(camera_img, (box[0, i], box[1, i]), (box[0, j], box[1, j]), color, thickness=thickness[1])
                    # top bottom
                    cv2.line(camera_img, (box[0, i + 4], box[1, i + 4]), (box[0, j + 4], box[1, j + 4]), color,
                             thickness=thickness[1])
                    # sideline
                    cv2.line(camera_img, (box[0, i], box[1, i]), (box[0, i + 4], box[1, i + 4]), color,
                             thickness=thickness[1])
            else:
                min_x = min(box[0, :])
                min_y = min(box[1, :])
                max_x = max(box[0, :])
                max_y = max(box[1, :])
                cv2.rectangle(camera_img, (min_x, min_y), (max_x, max_y), color, thickness[1])


def show_projected_3D_bbox(nusc, sample, anns_info, show_2D):
    cv2.namedWindow("image window", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("image window", 1200, 1200)
    camera_file = dict()
    data_root = nusc.dataroot
    for key in sample['data']:
        if key.startswith('CAM'):
            sample_data = nusc.get('sample_data', sample['data'][key])
            camera_file[sample_data['channel']] = sample_data

    img_list = list()
    ori_img_size = (1600, 900)
    for camera_type in ['FRONT', 'FRONT_LEFT', 'FRONT_RIGHT', 'BACK', 'BACK_LEFT', 'BACK_RIGHT']:
        camera_data = camera_file['CAM_{}'.format(camera_type)]
        img_path = os.path.join(data_root, camera_data['filename'])
        img = cv2.imread(img_path)
        calib_data = nusc.get('calibrated_sensor', camera_data['calibrated_sensor_token'])
        ego_data = nusc.get('ego_pose', camera_data['ego_pose_token'])
        obj3d_list = list()
        for ann in anns_info:
            obj = get_obj3d_from_annotation(ann, ego_data, calib_data)
            if obj is not None:
                obj3d_list.append(obj)
        obj2d_list = project_obj2image(obj3d_list, calib_data['camera_intrinsic'])
        plot_annotation_info(None, img, obj2d_list, show_2D)
        img_list.append(img)
    img_set1 = np.concatenate(img_list[:3], axis=0)
    img_set2 = np.concatenate(img_list[3:], axis=0)
    camera_img = np.concatenate((img_set1, img_set2), axis=1)
    del img, img_list, img_set1, img_set2
    img_h, img_w, _ = (np.asarray(camera_img.shape) * 0.37)
    camera_img = cv2.resize(camera_img, (int(img_w), int(img_h)))
    cv2.imshow('image window', camera_img)
    cv2.waitKey(0)


def project_3D_to_2D(nusc, anns_info, cam_sample_data, lidar_2D_bbox, cam_frame, channel_id):
    calib_data = nusc.get('calibrated_sensor', cam_sample_data['calibrated_sensor_token'])
    ego_data = nusc.get('ego_pose', cam_sample_data['ego_pose_token'])
    obj3d_list = list()
    for ann in anns_info:
        obj = get_obj3d_from_annotation(ann, ego_data, calib_data)
        if obj is not None:
            obj3d_list.append(obj)
    for ind, Box_3D in enumerate(obj3d_list):
        Box_3D = [Box_3D]
        obj2d = project_obj2image(Box_3D, calib_data['camera_intrinsic'])
        if obj2d:
            box = obj2d[0]['box']
            min_x = min(min(box[0, :]), 1600.)
            min_y = min(min(box[1, :]), 900.)
            max_x = max(max(box[0, :]), 0.)
            max_y = max(max(box[1, :]), 0.)
            final_coords = torch.from_numpy(np.float32([max(min_x, 0.), max(min_y, 0.), min(max_x, 1600.), min(max_y, 900.)]))
            x1, y1, x2, y2 = lidar_2D_bbox[ind, :]
            if cam_frame[ind, :] == 0 or (x2 - x1) * (y2 - y1) < (final_coords[2] - final_coords[0]) * (
                    final_coords[3] - final_coords[1]):
                # min_x, min_y, max_x, max_y = final_coords
                cam_frame[ind, :] = channel_id
                lidar_2D_bbox[ind, :] = final_coords

    return lidar_2D_bbox, cam_frame
