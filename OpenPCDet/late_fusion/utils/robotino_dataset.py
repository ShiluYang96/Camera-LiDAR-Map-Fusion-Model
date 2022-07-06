import numpy as np
import torch
import time
import numba
import pickle
import json
from torch.utils.data import Dataset
from pcdet.datasets.kitti.kitti_object_eval_python.eval import d3_box_overlap
from tqdm import tqdm
from pathlib import Path
import glob

# ignore the warning from numba
from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning,NumbaPerformanceWarning,NumbaWarning
import warnings
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPerformanceWarning)
warnings.simplefilter('ignore', category=NumbaWarning)
warnings.simplefilter('ignore')
warnings.filterwarnings('ignore')


class clocs_data(Dataset):

    def __init__(self, _2d_path, _3d_path, input_data, root_path, val=False):

        self._2d_path = _2d_path
        self._3d_path = _3d_path
        self.root_path = root_path
        self.sample_id_list = self.get_index()
        self.val = val
        self.input_data = input_data
        self.idx_list = self.sample_id_list['testing'] if self.val else self.sample_id_list['training']

    def get_index(self): # get training testing file id
        sample_id_list = {}
        for split in ['training', 'testing']:
            root_split_path = self.root_path + '/robotino/' + split
            points_file_list = glob.glob(str(root_split_path + '/points/*.pcd'))
            points_file_list.sort()
            # sample_label_file_list = labels_file_list
            sample_id_list[split] = [Path(points_file_list[index]).stem for index in range(len(points_file_list))]
        return sample_id_list

    def get_label(self, file_index, split):
        label_file = self.root_path + '/robotino/' + split + '/labels/' + file_index + '.json'
        with open(label_file, 'r') as f:
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
        return points_label

    def generate_input(self):
        # make dir for input tensor
        input_path = Path(self.input_data)
        if not input_path.exists():
            input_path.mkdir(parents=True)

        for split in ['training', 'testing']:
            for idx in tqdm(self.sample_id_list[split]):

                # find 2d detection
                detection_2d_file_name = self._2d_path + "/" + idx + ".txt"
                try:
                    with open(detection_2d_file_name, 'r') as f:
                        lines = f.readlines()
                except FileNotFoundError: # no 2D proposals
                    lines = ['Robotino -1 -1 -10 0 0.00 0 0 -1 -1 -1 -1000 -1000 -1000 -10 0 ']

                content = [line.strip().split(' ') for line in lines]
                predicted_class = np.array([x[0] for x in content], dtype='object')
                predicted_class_index = np.where(predicted_class == 'Robotino')

                # get bbox in 2d
                detection_result = np.array([[float(info) for info in x[4:8]] for x in content]).reshape(-1, 4)
                score = np.array([float(x[15]) for x in content])  # 1000 is the score scale!!!
                f_detection_result = np.append(detection_result, score.reshape(-1, 1), 1)    # ([x1,y1,x2,y2,score])
                middle_predictions = f_detection_result[predicted_class_index, :].reshape(-1, 5)  # score ranking
                top_predictions = middle_predictions[np.where(middle_predictions[:, 4] >= -100)]  # not do filter

                # get 3d result
                _3d_result = torch.load(self._3d_path + "/" + idx + ".pt")[0]

                # get input data
                res, iou_test, tensor_index = self.train_stage_2(_3d_result, top_predictions)
                all_3d_output_camera_dict, fusion_input, tensor_index = res, iou_test, tensor_index

                # get 3d anno
                d3_gt_boxes = self.get_label(idx, split)

                # get training label
                d3_gt_boxes_camera = d3_gt_boxes
                if d3_gt_boxes.shape[0] == 0:
                    target_for_fusion = np.zeros((1, 20000, 1))
                    positive_index = np.zeros((1, 20000), dtype=np.float32)
                    negative_index = np.zeros((1, 20000), dtype=np.float32)
                    negative_index[:, :] = 1
                else:
                    ###### predicted bev boxes
                    pred_3d_box = all_3d_output_camera_dict[0]["box3d_camera"]
                    iou_bev = d3_box_overlap(d3_gt_boxes_camera, pred_3d_box, criterion=-1)
                    iou_bev_max = np.amax(iou_bev, axis=0)
                    # print(np.max(iou_bev_max))
                    target_for_fusion = ((iou_bev_max >= 0.5) * 1).reshape(1, -1, 1)

                    positive_index = ((iou_bev_max >= 0.5) * 1).reshape(1, -1)
                    negative_index = ((iou_bev_max <= 0.25) * 1).reshape(1, -1)

                # save data and label
                all_data = {}
                all_data['input_data'] = {'fusion_input': fusion_input.numpy(), 'tensor_index': tensor_index.numpy()}
                all_data['label'] = {'target_for_fusion': torch.tensor(target_for_fusion),
                                     'positive_index': torch.tensor(positive_index),
                                     'negative_index': torch.tensor(negative_index),
                                     'label_n': len(d3_gt_boxes_camera)}
                torch.save(all_data, self.input_data + '/' + idx + '.pt')

    def train_stage_2(self, _3d_result, top_predictions):
        box_preds = _3d_result['boxes_lidar']  # (x,y,z,h,w,l,r)
        final_box_preds = box_preds
        predictions_dicts = []
        locs = _3d_result['location']
        dims = _3d_result['dimensions']
        angles = _3d_result['rotation_y'].reshape(-1, 1)
        final_box_preds_camera = np.concatenate((locs, dims, angles), axis=1)
        box_2d_preds = _3d_result['bbox']
        final_scores = _3d_result['score']
        img_idx = _3d_result['frame_id']
        # predictions
        predictions_dict = {
            "bbox": box_2d_preds,
            "box3d_camera": final_box_preds_camera,
            "box3d_lidar": final_box_preds,
            "scores": final_scores,
            # "label_preds": label_preds,
            "image_idx": img_idx,
        }
        predictions_dicts.append(predictions_dict)
        # get (x*x + y*y) / 5.7  --> distance between target and lidar (0, 1)
        dis_to_lidar = torch.norm(torch.tensor(box_preds[:, :2]), p=2, dim=1, keepdim=True).numpy() / 5.7

        box_2d_detector = np.zeros((200, 4))
        box_2d_detector[0:top_predictions.shape[0], :] = top_predictions[:, :4]
        box_2d_detector = top_predictions[:, :4]
        box_2d_scores = top_predictions[:, 4].reshape(-1, 1)

        time_iou_build_start = time.time()
        overlaps1 = np.zeros((900000, 4), dtype=np.float32)
        tensor_index1 = np.zeros((900000, 2), dtype=np.float32)
        overlaps1[:, :] = -1.0
        tensor_index1[:, :] = -1.0
        # final_scores[final_scores<0.1] = 0
        # box_2d_preds[(final_scores<0.1).reshape(-1),:] = 0
        iou_test, tensor_index, max_num = build_stage2_training(box_2d_preds,
                                                                box_2d_detector,
                                                                -1,
                                                                final_scores,
                                                                box_2d_scores,
                                                                dis_to_lidar,
                                                                overlaps1,
                                                                tensor_index1)
        time_iou_build_end = time.time()
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
        idx = self.idx_list[index]
        all_data = torch.load(self.input_data + '/' + idx + '.pt')
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
            return fusion_input, tensor_index, (self._3d_path + "/" + idx + ".pt")
        else:
            positives = positive_index.type(torch.float32)
            negatives = negative_index.type(torch.float32)
            one_hot_targets = target_for_fusion.type(torch.float32)
            return fusion_input, tensor_index, positives, negatives, one_hot_targets, label_n, idx

    def __len__(self):
        return len(self.idx_list)


# pang added to build the tensor for the second stage of training
@numba.jit(nopython=True, parallel=True)
def build_stage2_training(boxes, query_boxes, criterion, scores_3d, scores_2d, dis_to_lidar_3d, overlaps, tensor_index):
    N = boxes.shape[0]  # 20000  3d detector 2d bbox
    K = query_boxes.shape[0]  # 30  2d detector bbox
    max_num = 900000
    ind = 0
    ind_max = ind
    for k in range(K):
        qbox_area = ((query_boxes[k, 2] - query_boxes[k, 0]) *
                     (query_boxes[k, 3] - query_boxes[k, 1]))  # (x2-x1)*(y2-y1)
        for n in range(N):
            iw = (min(boxes[n, 2], query_boxes[k, 2]) -
                  max(boxes[n, 0], query_boxes[k, 0]))  # min(x2_3D, x2_2D)-max(x1_3D, x1_2D)
            if iw > 0:  # there is overlap in x axis
                ih = (min(boxes[n, 3], query_boxes[k, 3]) -
                      max(boxes[n, 1], query_boxes[k, 1]))  # check overlap on y axis
                if ih > 0:
                    if criterion == -1:
                        ua = (
                                (boxes[n, 2] - boxes[n, 0]) *
                                (boxes[n, 3] - boxes[n, 1]) + qbox_area - iw * ih)  # area_3d_bbox + area_2D_bbox - overlap
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
                    overlaps[ind, 0] = -10
                    overlaps[ind, 1] = scores_3d[n]
                    overlaps[ind, 2] = -10
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
    if ind > ind_max:
        ind_max = ind
    return overlaps, tensor_index, ind


if __name__ == '__main__':
    _2d_path='../data/2D_proposals'
    _3d_path = '../data/3D_proposals'
    root_path = '../data'
    input_data = root_path + '/input_tensor'
    train_dataset = clocs_data(_2d_path, _3d_path, input_data, root_path)
    train_dataset.generate_input()