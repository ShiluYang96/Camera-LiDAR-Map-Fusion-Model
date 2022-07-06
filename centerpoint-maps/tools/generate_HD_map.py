import math

import fire
from tqdm import tqdm

import cv2
import numpy as np
import PIL.Image as Image
from matplotlib import pyplot as plt
from nuscenes.map_expansion.map_api import NuScenesMap, NuScenesMapExplorer
from pyquaternion import Quaternion

try:
    from nuscenes import NuScenes, NuScenesExplorer
    from nuscenes.utils import splits
    from nuscenes.utils.data_classes import LidarPointCloud
    from nuscenes.utils.geometry_utils import transform_matrix
    from nuscenes.utils.data_classes import Box
    from nuscenes.eval.detection.config import config_factory
    from nuscenes.eval.detection.evaluate import NuScenesEval
except:
    print("nuScenes devkit not Found!")


def fig2data(fig):
    """
    fig = plt.figure()
    image = fig2data(fig)
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    image = Image.frombytes("RGBA", (w, h), buf.tobytes())
    image = np.asarray(image)
    return image


def crop_image(image: np.array,
               x_px: int,
               y_px: int,
               axes_limit_px: int) -> np.array:
    x_min = int(x_px - axes_limit_px)
    x_max = int(x_px + axes_limit_px)
    y_min = int(y_px - axes_limit_px)
    y_max = int(y_px + axes_limit_px)

    cropped_image = image[y_min:y_max, x_min:x_max]
    return cropped_image


def generate_HD_map(nusc, sample, sample_data_token, out_path):
    # get the location / name of map
    scene = nusc.get('scene', sample['scene_token'])
    log_token = scene['log_token']
    log = nusc.get('log', log_token)
    location = log['location']
    axes_limit = 40.0
    # load map
    nusc_map = NuScenesMap(dataroot='data/nuScenes', map_name=location)
    # get the ego pose of the corresponding sample
    sample_data_record = nusc.get('sample_data', sample_data_token)
    pose_record = nusc.get('ego_pose', sample_data_record['ego_pose_token'])
    ego_center_x, ego_center_y, ego_center_z = pose_record['translation']
    layer_names = ['drivable_area', 'walkway', 'carpark_area']
    # layer_names = ['road_segment', 'lane', 'ped_crossing', 'walkway', 'stop_line', 'carpark_area']

    # render map_patch
    axes_limit_px = int(axes_limit * math.sqrt(2))
    x_min, y_min, x_max, y_max = ego_center_x - axes_limit_px, ego_center_y - axes_limit_px, ego_center_x + axes_limit_px, ego_center_y + axes_limit_px
    patch_box = [x_min, y_min, x_max, y_max]
    fig, ax = nusc_map.render_map_patch(patch_box, layer_names,
                                        render_egoposes_range=False, render_legend=False)
    # transfer fig to img numpy array
    image = fig2data(fig)
    # alignment
    ypr_rad = Quaternion(pose_record['rotation']).yaw_pitch_roll
    yaw_deg = -math.degrees(ypr_rad[0])
    rotated_cropped = np.array(Image.fromarray(image).rotate(yaw_deg))
    # crop
    scaled_limit_px = 400
    ego_centric_map = crop_image(rotated_cropped, rotated_cropped.shape[1] / 2,
                                 rotated_cropped.shape[0] / 2,
                                 scaled_limit_px)
    # save map data
    cv2.imwrite(out_path, ego_centric_map)
    # clear plots
    f = plt.figure()
    f.clear()
    plt.close("all")


def create_HDmap(root_path, version, nsweeps=1, raw=True):
    nusc = NuScenes(version=version, dataroot=root_path, verbose=True)
    chan = "LIDAR_TOP"  # The reference channel of the current sample_rec that the point clouds are mapped to.

    for sample in tqdm(nusc.sample):
        sample_data_token = sample["data"][chan]

        # read map data ans save with token name
        out_path = "data/nuScenes/maps_generated/HDMAP/" + sample_data_token + "__HDMAP.png"
        if raw:  # save raw map data with lidar
            nusc.render_sample_data(sample_data_token, with_anns=False, nsweeps=nsweeps, out_path=out_path, underlay_map=True)
            f = plt.figure()
            f.clear()
            plt.close("all")
        else:
            generate_HD_map(nusc, sample, sample_data_token, out_path)


if __name__ == '__main__':
    fire.Fire()

    """
    nusc = NuScenes(version="v1.0-trainval", dataroot="data/nuScenes", verbose=False)
    nusc_exp = NuScenesExplorer(nusc)

    # layer_names = ['road_segment', 'lane', 'ped_crossing', 'walkway', 'stop_line', 'carpark_area']

    my_scene = nusc.scene[0]
    first_sample_token = my_scene['first_sample_token']
    my_sample = nusc.get('sample', first_sample_token)
    sample_data_token = my_sample['data']['LIDAR_TOP']

    # out_path = f"data/nuScenes/maps_generated/{sample_data_token}"
    out_path = "data/nuScenes/maps_generated/" + sample_data_token + ".png"
    nusc.render_sample_data(sample_data_token, with_anns=False, nsweeps=10, out_path=None, underlay_map=True)
    # if out_path is not None:
    #    plt.savefig(out_path, bbox_inches='tight', pad_inches=0)

    generate_HD_map(nusc, my_sample, sample_data_token, out_path)
    plt.show()
    
    nusc_exp.render_ego_centric_map(sample_data_token, axes_limit=40)
    if out_path is not None:
        plt.savefig(out_path)
    plt.show()
    """
