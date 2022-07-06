# -------------------------------------------------------------------------------------------------#
#   Although kmeans clusters the anchors in the dataset，many datastes are similar in size to 9 anchors
#   that are clustered，such nchors are not conducive to the training of the model
#   Because different feature layers are suitable for different sizes of a priori anchors,
#   the shallower feature layers are suitable for larger priori anchors.
#   The priori anchors of the original network has been allocated according to the proportion of
#   large, medium and small, and it will have very good results without clustering.
# -------------------------------------------------------------------------------------------------#
import glob
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from PIL import Image

img_dir = "/home/yang/centerpoint_maps/data/nuScenes/samples"
annotation_file = '/home/yang/mmdetection/efficientDet/others/train.txt'
input_shape = [768, 768]
anchors_num = 3


def cas_iou(box, cluster):
    x = np.minimum(cluster[:, 0], box[0])
    y = np.minimum(cluster[:, 1], box[1])

    intersection = x * y
    area1 = box[0] * box[1]

    area2 = cluster[:,0] * cluster[:,1]
    iou = intersection / (area1 + area2 - intersection)

    return iou


def avg_iou(box, cluster):
    return np.mean([np.max(cas_iou(box[i], cluster)) for i in range(box.shape[0])])


def kmeans(box, k):
    # -------------------------------------------------------------#
    #   Get the number of anchors
    # -------------------------------------------------------------#
    row = box.shape[0]

    # -------------------------------------------------------------#
    #   The position of each point in each anchor
    # -------------------------------------------------------------#
    distance = np.empty((row, k))

    # -------------------------------------------------------------#
    #   The position after clustering
    # -------------------------------------------------------------#
    last_clu = np.zeros((row,))

    np.random.seed()

    # -------------------------------------------------------------#
    #   Randomly choose 5 points as cluster-center
    # -------------------------------------------------------------#
    cluster = box[np.random.choice(row, k, replace=False)]

    iter = 0
    while True:
        # -------------------------------------------------------------#
        #   Calculate the iou of each line away from the five points
        # -------------------------------------------------------------#
        for i in range(row):
            distance[i] = 1 - cas_iou(box[i], cluster)

        # -------------------------------------------------------------#
        #   Get the point with min iou
        # -------------------------------------------------------------#
        near = np.argmin(distance, axis=1)

        if (last_clu == near).all():
            break

        # -------------------------------------------------------------#
        #   Find the midpoint of each class
        # -------------------------------------------------------------#
        for j in range(k):
            cluster[j] = np.median(
                box[near == j], axis=0)

        last_clu = near
        if iter % 5 == 0:
            print('iter: {:d}. avg_iou:{:.2f}'.format(iter, avg_iou(box, cluster)))
        iter += 1

    return cluster, near


def get_image_id(image_data):
    image_id = ((image_data.split(' ')[0]).split('/')[-1]).split('.')
    return image_id[0]


def get_image_path(image_data):
    image_path = (image_data.split(' ')[0])
    return image_path


def read_box_coco(path, img_dir):
    data = []
    image_data = open(path).read().strip().split('\n')
    image_paths = [get_image_path(image_data) for image_data in image_data]
    image_objs = [image_data.split(' ')[1:] for image_data in image_data]
    index = 0
    for image_path in tqdm(image_paths):
        # image = Image.open(image_path)
        # width, height = image.size
        width, height = 1600, 900
        objs = image_objs[index]
        index = index + 1
        for obj in objs:
            obj = obj.split(',')
            # x_min, y_min, x_max, y_max, int(info[1]
            left = np.float64(obj[0]) / width
            bottom = np.float64(obj[1]) / height
            right = np.float64(obj[2]) / width
            top = np.float64(obj[3]) / height
            data.append([right - left, top - bottom])
    return np.array(data)


def compute_anchor_para(bboxs, anchor_base_scale = 4, anchor_stride = 8):
    """
    Compute anchor parameters, given all bboxes from kmean gathered
    Require anchor_base_scale, anchor_stride at first feature map, it depends on network configuration
    return anchor scale and anchor ratios
    default parameter should work for Resnet50 backbone
    """
    return_scale, return_ratio = [], []
    base_factor = anchor_base_scale * anchor_stride
    for height, width in bboxs:
        return_scale.append(height*1.0/base_factor)
        return_ratio.append((1,width*1.0/height))
    return return_scale, return_ratio


if __name__ == '__main__':

    np.random.seed(0)
    np.seterr(divide='ignore', invalid='ignore')

    # -------------------------------------------------------------#
    #   Load all image annotation
    # -------------------------------------------------------------#
    print('Load annotation.')
    data = read_box_coco(annotation_file, img_dir)
    print('Load annotation done.')
    # -------------------------------------------------------------#
    #   Use Kmeans clustering
    # -------------------------------------------------------------#
    print('K-means boxes.')
    cluster, near = kmeans(data, anchors_num)
    print('K-means boxes done.')
    data = data * np.array([input_shape[1], input_shape[0]])
    cluster = cluster * np.array([input_shape[1], input_shape[0]])

    cluster = cluster[np.argsort(cluster[:, 0] * cluster[:, 1])]
    print('avg_ratio:{:.2f}'.format(avg_iou(data, cluster)))
    print(cluster)
    print("computed paras: ", compute_anchor_para(cluster))
