"""
The official account of Lele perception school
@author: https://blog.csdn.net/suiyingy
modified: S. Y
"""
import re
import os
from mayavi import mlab
import json
import open3d as o3d
import numpy as np
import math
import argparse

# coordinate transfer
def pixelPlace(theta, distance):
    # 240 - 480 --> -120 - 120
    theta = theta - 360
    # crop the laserscan in camera view
    if -36 <= theta <= 36:
        x = distance * np.cos(theta * np.pi / 180)
        y = distance * np.sin(theta * np.pi / 180)
        z = 0.3
        intensity = 1.0
    else:
        x = 0.
        y = 0.
        z = 0.
        intensity = 0.
    return x, y, z, intensity


def process_ranges(ranges_list, angle_min, angle_increment):
    # radian --> degree
    angle_min = angle_min / np.pi * 180
    angle_increment = angle_increment / np.pi * 180
    ranges_array = np.zeros((len(ranges_list), 4))
    for id, range in enumerate(ranges_list):
        range = float(range)
        angle = angle_min + angle_increment * id
        x, y, z, intensity = pixelPlace(angle, range)
        ranges_array[id,:] = [x, y, z, intensity]
        # ranges_array[id, :] = [range, angle, 0]
    return ranges_array


# visualize point clouds
def viz_mayavi(points, vals="distance"):
    x = points[:, 0]  # x position of point
    y = points[:, 1]  # y position of point
    z = points[:, 2]  # z position of point
    fig = mlab.figure(bgcolor=(0, 0, 0), size=(1280, 720))
    # d=torch.sqrt(x**2+y**2)
    d = np.sqrt(x ** 2 + y ** 2)
    if vals=="height":
        col=z
    else:
        col=d
    mlab.points3d(x,y,z,
                         col,
                         mode="point",
                         colormap='spectral',
                         figure=fig,
                         )
    mlab.show()


if  __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", help="Path of dataset root", type=str,
                        default='/home/foj-sy/Downloads/dataset/Robotino')
    parser.add_argument("--out_path", help="output dir", type=str, default="lidar")
    args = parser.parse_args()

    # data path
    data_path = args.data_path
    out_path = args.out_path
    # file_name = './velodyne/000000.bin'
    # points = np.fromfile(file_name, dtype=np.float32).reshape([-1, 4])
    # points = np.load(file_name)
    # viz_mayavi(points)

    # pcd = o3d.io.read_point_cloud("lidar/20220530-100830.pcd")
    # out_arr = np.asarray(pcd.points, dtype=np.float32)

    os.makedirs(out_path,exist_ok=True)
    # iterate all files in path
    for filename in os.listdir(data_path):
        # check file ending
        if filename.endswith('.txt'):
            # get file name
            file_name_ros = os.path.join(data_path, filename)
            time_stamp = filename.split('.')[0]
            out_file = os.path.join(out_path, time_stamp + '.pcd')
            # read file
            with open(file_name_ros, 'r') as f:
                lines = f.readlines()[-1]

            ranges = lines.strip()[1:-1]
            ranges = ranges.split(', ')[0:-1]
            angle_min = 4.188789367675781
            angle_increment = 0.00872664526104927
            ranges_array = process_ranges(ranges, angle_min, angle_increment)
            # remove all 0. row
            ranges_array = ranges_array[~np.all(ranges_array == 0., axis=1)]
            # viz_mayavi(ranges_array)
            # np.save(out_file, ranges_array)

            # save points in pcd file
            xyz = ranges_array[:, 0:3]
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz)
            o3d.io.write_point_cloud(out_file, pcd)


    """
    # point_clouds = np.empty([0, 3])
    for line in lines:
        if 'angle_min' in line:
            angle_min = line.strip().split('angle_min: ')[-1]
            angle_min = float(angle_min)
        elif 'angle_increment: ' in line:
            angle_increment = line.strip().split('angle_increment: ')[-1]
            angle_increment = float(angle_increment)
        elif 'ranges:' in line:
            ranges = line.strip().split('ranges: [')[-1]
            ranges = re.split(', |]', ranges)[0:-1]
            ranges_array = process_ranges(ranges, angle_min, angle_increment)
            # remove all 0. row
            ranges_array = ranges_array[~np.all(ranges_array == 0, axis=1)]
            point_clouds = np.vstack((point_clouds,ranges_array))
            viz_mayavi(ranges_array)
    viz_mayavi(point_clouds)
    # save array in bin file
    np.save(out_file, point_clouds)
    # points = np.load(out_file)
    # viz_mayavi(points)
    """