import os.path

import cv2
import numpy as np
import torch
import numba
import pickle

from nuscenes import NuScenes
from nuscenes.utils import splits
from pyquaternion import Quaternion
from torch.utils.data import Dataset
from tqdm import tqdm
from pathlib import Path
from det3d.datasets.kitti.eval import d3_box_overlap
from det3d.datasets.nuscenes.nusc_common import _second_det_to_nusc_box, _lidar_nusc_box_to_global
from det3d.datasets.nuscenes.nusc_common_map import quaternion_yaw, _get_available_scenes
from late_fusion.export_3D_to_2D import project_3D_to_2D, show_projected_3D_bbox

# ignore the warning from numba
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning,NumbaPerformanceWarning,NumbaWarning
import warnings
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)
warnings.simplefilter('ignore', category=NumbaWarning)
warnings.simplefilter('ignore')
warnings.filterwarnings('ignore')


class Late_fusion_dataset(Dataset):

    def __init__(self, nusc, root_path, _2d_path, _3d_path, index_path, input_data,
                 info_path_train, info_path_val, val=False):
        self.nusc = nusc
        self._2d_path = _2d_path
        self._3d_path = _3d_path
        self.root_path = root_path
        try:
            with open(index_path, "r") as f:
                self.ind = f.read().splitlines()
            if len(self.ind) == 0:
                raise FileNotFoundError
        except FileNotFoundError:
            print("There is no index file! Generating....")
            self.generate_index()
            with open(index_path, "r") as f:
                self.ind = f.read().splitlines()

        self.val = val
        self.input_data = input_data
        self.anno_train = pickle.load(open(info_path_train, 'rb'))
        self.anno_val = pickle.load(open(info_path_val, 'rb'))
        self.anno = self.anno_train + self.anno_val
        self.cam_channels = ["CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT", "CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT"]

        self.id2ind = np.empty(len(self.anno), dtype=object)
        for i in range(len(self.anno)):
            self.id2ind[i] = self.anno[i]["token"]
        # result = np.where(self.id2ind == "e93e98b63d3b40209056d129dc53ceee")

    def get_2d_file(self, sample, ref_chan):
        """
        input: sample
        output: detection results from ref. camera channel
        """
        # get result file .txt
        cam_token = sample["data"][ref_chan]
        orig_cam_path, _, ref_cam_intrinsic = self.nusc.get_sample_data(cam_token)
        cam_path = orig_cam_path.split('/')[-1]
        cam_path = cam_path.split('.')[0]
        file_name_2D = ref_chan + "/" + cam_path + ".txt"
        detection_2d_file_name = os.path.join(self._2d_path, file_name_2D)
        with open(detection_2d_file_name, 'r') as f:
            lines = f.readlines()

        content = [line.strip().split(' ') for line in lines]
        predicted_class = np.array([x[0] for x in content], dtype='object')
        predicted_class_index = np.where(predicted_class == 'Car')
        detection_result = np.array([[float(info) for info in x[4:8]] for x in content], dtype=np.float32).reshape(-1, 4)  # 2d bbox
        score = np.array([float(x[15]) for x in content], dtype=np.float32) # 1000 is the score scale!!!
        detection_result = torch.from_numpy(detection_result).cuda()
        score = torch.from_numpy(score).cuda()
        f_detection_result = torch.cat([detection_result, score.reshape(-1, 1)], dim=1)  # ([x1,y1,x2,y2,score])

        # middle_predictions = f_detection_result[predicted_class_index, :].reshape(-1, 5)  # score ranking
        # top_predictions = middle_predictions[np.where(middle_predictions[:, 4] >= -100)]  # not do filter
        return f_detection_result, orig_cam_path

    def show_bbox(self, img_full_name, left, right, top, bottom):
        # load the image and scale it
        image = cv2.imread(img_full_name)
        thick = 3
        color = (0, 0, 255)
        cv2.namedWindow("image window", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("image window", 1200, 1200)

        cv2.rectangle(image, (int(left), int(top)), (int(right), int(bottom)), color, thick)

        cv2.imshow('image window', image)
        # add wait key. window waits until user presses a key
        cv2.waitKey(3000)
        # and finally destroy/close all open windows
        cv2.destroyAllWindows()

    def process_anno(self, anno, class_name=['car']):
        ind = np.where(anno['gt_names'] == class_name[0])
        gt_boxes = anno['gt_boxes'][ind]  # [locs, dims, velocity[:, :2], -rots - np.pi / 2]
        loc = gt_boxes[:, :3]  # x y z
        dim = gt_boxes[:, 3:6]  # w l h
        rot = gt_boxes[:, -1]
        if len(loc) == 0:
            d3_box = []
        else:
            d3_box = np.concatenate((loc, dim, rot.reshape(-1, 1)), axis=1)
        return np.array(d3_box)  # (x,y,z,w,l,h,r)

    def generate_input(self):
        input_path = Path(self.input_data)
        if not input_path.exists():
            input_path.mkdir(parents=True)

        for sample in tqdm(self.nusc.sample):
            sample_token = sample['token']

            # get 3d results
            File_name_3D = self._3d_path + "/" + sample_token + ".pt"
            predict_3d = torch.load(File_name_3D)
            lidar_2D_bbox = torch.zeros([predict_3d["box3d_lidar"].shape[0],4], dtype=torch.float32).cuda()
            cam_frame = torch.zeros((predict_3d["box3d_lidar"].shape[0],1), dtype=torch.int32).cuda()

            # transform 3d lidar boxes to global coordinate
            boxes = _second_det_to_nusc_box(predict_3d)
            boxes_3d_global = _lidar_nusc_box_to_global(self.nusc, boxes, predict_3d["metadata"]["token"])
            Cam_boxes = {}
            # CAM_paths = []
            for id, cam_channel in enumerate(self.cam_channels):
                channel_id = id + 1
                cam_sample_data = self.nusc.get('sample_data', sample['data'][cam_channel])
                # project lidar 3d boxes in image coordinate
                lidar_2D_bbox, cam_frame = project_3D_to_2D(self.nusc, boxes_3d_global, cam_sample_data, lidar_2D_bbox, cam_frame, channel_id)  # [2d_boxes],[channel_id]
                # get camera_box
                Cam_box, cam_path = self.get_2d_file(sample, cam_channel)  # [[x1,y1,x2,y2,score]..]
                Cam_boxes[cam_channel] = Cam_box
                # CAM_paths.append(cam_path)

            # show the projected 3D box
            # show_projected_3D_bbox(self.nusc, sample, boxes_3d_global, show_2D=False)

            # show the projected 2D box
            # for i in range(lidar_2D_bbox.size(dim=0)):
            #     x1, y1, x2, y2 = lidar_2D_bbox[i, :]
            #     cam_path = CAM_paths[cam_frame[i,:]-1]
            #     self.show_bbox(cam_path, x1, x2, y2, y1)

            predict_3d["bbox"] = lidar_2D_bbox
            predict_3d["cam_frame"] = cam_frame

            # get input data
            res, iou_test, tensor_index = self.train_stage_2(predict_3d, Cam_boxes)
            all_3d_output_camera_dict, fusion_input, tensor_index = res, iou_test, tensor_index

            # get 3d anno
            int_ind = np.where(self.id2ind == sample_token)
            int_ind = int(int_ind[0])
            gt_anno = self.anno[int_ind]
            d3_gt_boxes = self.process_anno(gt_anno)

            # get training label
            d3_gt_boxes_camera = d3_gt_boxes
            if d3_gt_boxes.shape[0] == 0:
                target_for_fusion = np.zeros((1, 20000, 1))
                positive_index = np.zeros((1, 20000), dtype=np.float32)
                negative_index = np.zeros((1, 20000), dtype=np.float32)
                negative_index[:, :] = 1
            else:
                pred_3d_box = all_3d_output_camera_dict[0]["box3d_camera"].detach().cpu().numpy()
                iou_bev = d3_box_overlap(d3_gt_boxes_camera, pred_3d_box, criterion=-1)
                iou_bev_max = np.amax(iou_bev, axis=0)
                # print(np.max(iou_bev_max))
                target_for_fusion = ((iou_bev_max >= 0.5) * 1).reshape(1, -1, 1)  # for nuScenes thres = 0.5

                positive_index = ((iou_bev_max >= 0.5) * 1).reshape(1, -1)
                negative_index = ((iou_bev_max <= 0.25) * 1).reshape(1, -1)

            # save data and label
            all_data = {}
            all_data['input_data'] = {'fusion_input': fusion_input.numpy(), 'tensor_index': tensor_index.numpy()}
            all_data['label'] = {'target_for_fusion': torch.tensor(target_for_fusion),
                                 'positive_index': torch.tensor(positive_index),
                                 'negative_index': torch.tensor(negative_index),
                                 'label_n': len(d3_gt_boxes_camera)}
            torch.save(all_data, self.input_data + '/' + sample_token + '.pt')

    def train_stage_2(self, predict_3d, Cam_boxes):
        box3d = predict_3d["box3d_lidar"].cuda()    # x, y, z, w, l, h, velocity, velocity, radians
        cam_frame = predict_3d["cam_frame"].detach().cpu().numpy()
        predictions_dicts = []

        locs = box3d[:, :6]  # x y z w l h
        quat = [Quaternion(axis=[0, 0, 1], radians=i) for i in box3d[:, -1]]
        rots = torch.Tensor([quaternion_yaw(b) for b in quat]).cuda().reshape(-1, 1)
        rots = -rots - np.pi / 2
        final_box_preds_camera = torch.cat([locs, rots], dim=1)

        box_2d_preds = predict_3d['bbox'].cpu().detach().numpy()
        final_scores = predict_3d['scores'].cpu().detach().numpy()
        token = predict_3d["metadata"]["token"]
        # predictions
        predictions_dict = {
            "bbox": predict_3d['bbox'].cpu(),
            "box3d_camera": final_box_preds_camera.cpu(),
            "box3d_lidar": box3d.cpu(),
            "scores": predict_3d['scores'],
            # "label_preds": label_preds,
            "token": token,
        }
        predictions_dicts.append(predictions_dict)
        # get (x*x + y*y) / predict_range  --> distance between target and lidar
        dis_to_lidar = torch.norm(box3d[:, :2], p=2, dim=1, keepdim=True).detach().cpu().numpy() / 72.4

        # process camera 2d boxes --> box and scores
        Box_2d_detector = torch.empty((0, 4), dtype=torch.float32).cuda()
        Box_2d_scores = torch.empty((0, 1), dtype=torch.float32).cuda()
        for cam_channel in self.cam_channels:
            cam_box = Cam_boxes[cam_channel]
            box_2d_detector = torch.zeros([100, 4], dtype=torch.float32).cuda()
            box_2d_scores = torch.zeros([100, 1], dtype=torch.float32).cuda()
            box_2d_detector[0:cam_box.shape[0], :] = cam_box[:, :4]
            box_2d_scores[0:cam_box.shape[0], :] = cam_box[:, 4].reshape(-1, 1)
            Box_2d_detector = torch.vstack([Box_2d_detector, box_2d_detector])
            Box_2d_scores = torch.vstack([Box_2d_scores, box_2d_scores])

        # time_iou_build_start = time.time()
        overlaps1 = np.zeros((900000, 4), dtype=np.float32)
        tensor_index1 = np.zeros((900000, 2), dtype=np.float32)
        overlaps1[:, :] = -1.0
        tensor_index1[:, :] = -1.0
        # final_scores[final_scores<0.1] = 0
        # box_2d_preds[(final_scores<0.1).reshape(-1),:] = 0

        iou_test, tensor_index, max_num = build_stage2_training(box_2d_preds,
                                                                Box_2d_detector.detach().cpu().numpy(),
                                                                -1,
                                                                final_scores,
                                                                Box_2d_scores.detach().cpu().numpy(),
                                                                dis_to_lidar,
                                                                overlaps1,
                                                                tensor_index1,
                                                                cam_frame)

        # time_iou_build_end = time.time()
        iou_test_tensor = torch.FloatTensor(iou_test)  # iou_test_tensor shape: [160000,4]
        tensor_index_tensor = torch.LongTensor(tensor_index)
        iou_test_tensor = iou_test_tensor.permute(1, 0)
        iou_test_tensor = iou_test_tensor.reshape(4, 900000)
        tensor_index_tensor = tensor_index_tensor.reshape(-1, 2)
        if max_num == 0:
            non_empty_iou_test_tensor = torch.zeros(4, 2)
            non_empty_iou_test_tensor[:, :] = -1
            non_empty_tensor_index_tensor = torch.zeros(2, 2)
            non_empty_tensor_index_tensor[:, :] = -1
        else:
            non_empty_iou_test_tensor = iou_test_tensor[:, :max_num]  # iou results of 2D bbox comparison
            non_empty_tensor_index_tensor = tensor_index_tensor[:max_num, :]  # no empty lidar proposals

        return predictions_dicts, non_empty_iou_test_tensor, non_empty_tensor_index_tensor

    def __getitem__(self, index):
        sample_token = self.ind[index]
        all_data = torch.load(self.input_data + '/' + sample_token + '.pt')
        inpu_data = all_data['input_data']
        label = all_data['label']
        fusion_input = torch.tensor(inpu_data['fusion_input']).reshape(4, 1, -1)
        tensor_index = torch.tensor(inpu_data['tensor_index']).reshape(-1, 2)
        target_for_fusion = label['target_for_fusion'].reshape(-1, 1)
        positive_index = label['positive_index'].reshape(-1)
        negative_index = label['negative_index'].reshape(-1)
        label_n = label['label_n']

        if self.val:
            # get 3d result
            return fusion_input, tensor_index, (self._3d_path + "/" + sample_token + ".pt")
        else:
            positives = positive_index.type(torch.float32)
            negatives = negative_index.type(torch.float32)
            one_hot_targets = target_for_fusion.type(torch.float32)
            return fusion_input, tensor_index, positives, negatives, one_hot_targets, label_n, sample_token

    def __len__(self):
        return len(self.ind)

    def generate_index(self):
        train_scenes = splits.train
        val_scenes = splits.val
        available_scenes = _get_available_scenes(self.nusc)
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

        os.makedirs(self.root_path + '/index/', exist_ok=True)
        out_path_train_index = self.root_path + '/index/train.txt'
        out_path_val_index = self.root_path + '/index/val.txt'

        with open(out_path_train_index, 'w') as f_train:
            with open(out_path_val_index, 'w') as f_val:
                for sample in tqdm(self.nusc.sample):
                    if sample["scene_token"] in train_scenes:
                        f_train.write(sample["token"])
                        f_train.write('\n')
                    elif sample["scene_token"] in val_scenes:
                        f_val.write(sample["token"])
                        f_val.write('\n')
        print("Success generated index info.")


@numba.jit(nopython=True, parallel=True)
def build_stage2_training(boxes, Box_2d_detector, criterion, scores_3d, Box_2d_scores, dis_to_lidar_3d, overlaps, tensor_index, cam_frame):

    N = boxes.shape[0]  # 20000  3d detector 2d bbox
    K = Box_2d_detector.shape[0] / 6  # 30  2d detector bbox
    max_num = 900000
    ind = 0
    ind_max = ind
    for n in range(N):
        # get the camera results from the corresponding camera channel
        cam_channel_id = (cam_frame[n,0] - 1)*100
        scores_2d = Box_2d_scores[cam_channel_id:(cam_channel_id + 100), :]
        query_boxes = Box_2d_detector[cam_channel_id:(cam_channel_id + 100), :]

        for k in range(K):
            if cam_channel_id >= 0:  # for 3d proposals with 2d projection
                iw = (min(boxes[n, 2], query_boxes[k, 2]) -
                      max(boxes[n, 0], query_boxes[k, 0]))  # min(x2_3D, x2_2D)-max(x1_3D, x1_2D)

                qbox_area = ((query_boxes[k, 2] - query_boxes[k, 0]) *
                             (query_boxes[k, 3] - query_boxes[k, 1]))  # (x2-x1)*(y2-y1)
                if iw > 0:  # there is overlap in x axis
                    ih = (min(boxes[n, 3], query_boxes[k, 3]) -
                          max(boxes[n, 1], query_boxes[k, 1]))  # check overlap on y axis
                    if ih > 0:
                        if criterion == -1:
                            ua = (
                                    (boxes[n, 2] - boxes[n, 0]) *
                                    (boxes[n, 3] - boxes[
                                        n, 1]) + qbox_area - iw * ih)  # area_3d_bbox + area_2D_bbox - overlap
                        elif criterion == 0:
                            ua = ((boxes[n, 2] - boxes[n, 0]) *
                                  (boxes[n, 3] - boxes[n, 1]))
                        elif criterion == 1:
                            ua = qbox_area
                        else:
                            ua = 1.0
                        overlaps[ind, 0] = iw * ih / ua  # IoU = overlap / (area_sum - overlap)
                        overlaps[ind, 1] = scores_3d[n]
                        overlaps[ind, 2] = scores_2d[k, 0]
                        overlaps[ind, 3] = dis_to_lidar_3d[n, 0]
                        tensor_index[ind, 0] = k
                        tensor_index[ind, 1] = n
                        ind = ind + 1

                    elif k == K - 1:
                        overlaps[ind, 0] = -10  #
                        overlaps[ind, 1] = scores_3d[n]
                        overlaps[ind, 2] = -10  #
                        overlaps[ind, 3] = dis_to_lidar_3d[n, 0]
                        tensor_index[ind, 0] = k
                        tensor_index[ind, 1] = n
                        ind = ind + 1
                elif k == K - 1:
                    overlaps[ind, 0] = -10
                    overlaps[ind, 1] = scores_3d[n]
                    overlaps[ind, 2] = -10
                    overlaps[ind, 3] = dis_to_lidar_3d[n, 0]
                    tensor_index[ind, 0] = k
                    tensor_index[ind, 1] = n
                    ind = ind + 1  # input tensor dim
            elif k == K - 1:
                overlaps[ind, 0] = -10
                overlaps[ind, 1] = scores_3d[n]
                overlaps[ind, 2] = -10
                overlaps[ind, 3] = dis_to_lidar_3d[n, 0]
                tensor_index[ind, 0] = k
                tensor_index[ind, 1] = n
                ind = ind + 1  # input tensor dim
    if ind > ind_max:
        ind_max = ind
    return overlaps, tensor_index, ind


if __name__ == '__main__':
    _2d_path = '/home/yang/centerpoint_maps/data/Output_2D'
    _3d_path = '/home/yang/centerpoint_maps/data/output_3d'
    root_path = '/home/yang/centerpoint_maps/data/nuScenes'
    input_data = root_path + '/modify_0'
    version = "v1.0-trainval"
    nusc = NuScenes(version=version, dataroot=root_path, verbose=True)

    train_ind_path = root_path + '/index/train.txt'
    val_ind_path = root_path + '/index/val.txt'

    info_path_train = root_path + '/infos_train_10sweeps_withvelo_filter_True.pkl'
    info_path_val = root_path + '/infos_val_10sweeps_withvelo_filter_True.pkl'
    train_dataset = Late_fusion_dataset(nusc, root_path, _2d_path, _3d_path, val_ind_path, input_data, info_path_train, info_path_val)
    train_dataset.generate_input()
