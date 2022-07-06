import itertools
import logging

from det3d.builder import build_box_coder
from det3d.utils.config_tool import get_downsample_factor

"""
We use the same config as nusc_centerpoint_pp_02voxel_circle_nms.py, except with the 
necessary modifications needed to use HD Maps 

TODO:

1. Verify the number of channels that our map net adds. Current assumption is 3 (rgb)
2. train_anno ; val_anno .pkl file names to be changed. - Not needed afaik
3. Check if we'll need to pass some checkpoint here/ Alternately, where do we load the saved model from.  - DONE

Later TODOs:
1. Check version (change to trainval if required for final full fledged training)
"""

norm_cfg = None

tasks = [
    dict(num_class=1, class_names=["car"]),
    # dict(num_class=2, class_names=["truck", "construction_vehicle"]),
    # dict(num_class=2, class_names=["bus", "trailer"]),
    # dict(num_class=1, class_names=["barrier"]),
    # dict(num_class=2, class_names=["motorcycle", "bicycle"]),
    # dict(num_class=2, class_names=["pedestrian", "traffic_cone"]),
]

class_names = list(itertools.chain(*[t["class_names"] for t in tasks]))

# training and testing settings
target_assigner = dict(
    tasks=tasks,
)

# model settings
model = dict(
    type="MapPP",
    pretrained=None,
    reader=dict(
        type="PillarFeatureNet",
        num_filters=[64],
        num_input_features=5,
        with_distance=False,
        voxel_size=(0.2, 0.2, 8),
        pc_range=(-51.2, -51.2, -5.0, 51.2, 51.2, 3.0),
        norm_cfg=norm_cfg,
    ),
    backbone=dict(type="PointPillarsScatter", ds_factor=1, norm_cfg=norm_cfg,),
    neck=dict(
        type="RPN",
        layer_nums=[3, 5, 5],
        ds_layer_strides=[2, 2, 2],
        ds_num_filters=[64, 128, 256],
        us_layer_strides=[0.5, 1, 2],
        us_num_filters=[128, 128, 128],
        num_input_features=64,
        norm_cfg=norm_cfg,
        logger=logging.getLogger("RPN"),
    ),
    map_net=dict(
        type="MapFusionNet",
    ),
    bbox_head=dict(
        # type='RPNHead',
        type="CenterHead",
        mode="3d",
        in_channels=sum([128, 128, 128, 64]),  # Mapnet will add 64 channels containing map features with same shape (128x128) as the RPN output
        norm_cfg=norm_cfg,
        tasks=tasks,
        dataset='nuscenes',
        weight=0.25,
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2, 1.0, 1.0],
        common_heads={'reg': (2, 2), 'height': (1, 2), 'dim':(3, 2), 'rot':(2, 2), 'vel': (2, 2)}, # (output_channel, num_conv)
        encode_rad_error_by_sin=False,
        direction_offset=0.0,
        bn=True
    ),
)

assigner = dict(
    target_assigner=target_assigner,
    out_size_factor=get_downsample_factor(model),
    dense_reg=1,
    gaussian_overlap=0.1,
    max_objs=500,
    min_radius=2,
)


train_cfg = dict(assigner=assigner)


# test_cfg = dict(
#     post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
#     max_per_img=500,
#     max_pool_nms=False,
#     min_radius=[4, 12, 10, 1, 0.85, 0.175],
#     post_max_size=83,
#     score_threshold=0.1,
#     pc_range=[-51.2, -51.2],
#     out_size_factor=get_downsample_factor(model),
#     voxel_size=[0.2, 0.2],
#     circle_nms=True
# )
test_cfg = dict(
    post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
    max_per_img=5000,  # change
    max_pool_nms=False,
    # min_radius=[4, 12, 10, 1, 0.85, 0.175],
    min_radius=[0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001],  # for nms distance threshold
    post_max_size=1000,  # change
    score_threshold=0.01,  # set a 0 or very small scores for proposals generation
    pc_range=[-51.2, -51.2],  # prediction range
    out_size_factor=get_downsample_factor(model),
    voxel_size=[0.2, 0.2],
    circle_nms=True,  # deactivate
    nms=dict(
        nms_pre_max_size=5000,  #
        nms_post_max_size=1000,  # number of output proposals
        nms_iou_threshold=0.001,  #
        use_rotate_nms=False,
        use_multi_class_nms=False
    )
)

# dataset settings
dataset_type = "NuScenesDataset"
nsweeps = 10
data_root = "data/nuScenes"

db_sampler = dict(
    type="GT-AUG",
    enable=False,
    db_info_path="data/nuScenes/dbinfos_train_10sweeps_withvelo.pkl",
    sample_groups=[
        dict(car=2),
        # dict(truck=3),
        # dict(construction_vehicle=7),
        # dict(bus=4),
        # dict(trailer=6),
        # dict(barrier=2),
        # dict(motorcycle=6),
        # dict(bicycle=6),
        # dict(pedestrian=2),
        # dict(traffic_cone=2),
    ],
    db_prep_steps=[
        dict(
            filter_by_min_num_points=dict(
                car=5,
                # truck=5,
                # bus=5,
                # trailer=5,
                # construction_vehicle=5,
                # traffic_cone=5,
                # barrier=5,
                # motorcycle=5,
                # bicycle=5,
                # pedestrian=5,
            )
        ),
        dict(filter_by_difficulty=[-1],),
    ],
    global_random_rotation_range_per_object=[0, 0],
    rate=1.0,
)
train_preprocessor = dict(
    mode="train",
    shuffle_points=True,
    gt_loc_noise=[0.0, 0.0, 0.0],
    gt_rot_noise=[0.0, 0.0],
    global_rot_noise=[-0.3925, 0.3925],
    global_scale_noise=[0.95, 1.05],
    global_rot_per_obj_range=[0, 0],
    global_trans_noise=[0.2, 0.2, 0.2],
    remove_points_after_sample=False,
    gt_drop_percentage=0.0,
    gt_drop_max_keep_points=15,
    remove_unknown_examples=False,
    remove_environment=False,
    db_sampler=db_sampler,
    class_names=class_names,
)

val_preprocessor = dict(
    mode="val",
    shuffle_points=False,
    remove_environment=False,
    remove_unknown_examples=False,
)

voxel_generator = dict(
    range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
    voxel_size=[0.2, 0.2, 8],
    max_points_in_voxel=20,
    max_voxel_num=30000,
)

train_pipeline = [
    dict(type="LoadPointCloudFromFile", dataset=dataset_type),
    dict(type="LoadPointCloudAnnotations", with_bbox=True),
    dict(type="Preprocess", cfg=train_preprocessor),
    dict(type="Voxelization", cfg=voxel_generator),
    dict(type="AssignLabel", cfg=train_cfg["assigner"]),
    dict(type="LoadRasterizedMapFromFile"),
    dict(type="ReformatWithRasterizedMap"),
]
test_pipeline = [
    dict(type="LoadPointCloudFromFile", dataset=dataset_type),
    dict(type="LoadPointCloudAnnotations", with_bbox=True),
    dict(type="Preprocess", cfg=val_preprocessor),
    dict(type="Voxelization", cfg=voxel_generator),
    dict(type="AssignLabel", cfg=train_cfg["assigner"]),
    dict(type="LoadRasterizedMapFromFile"),
    dict(type="ReformatWithRasterizedMap"),
]

train_anno = "data/nuScenes/infos_train_10sweeps_withvelo_filter_True.pkl"
val_anno = "data/nuScenes/infos_val_10sweeps_withvelo_filter_True.pkl"
# val_anno = "data/nuScenes/infos_train_10sweeps_withvelo_filter_True.pkl"
test_anno = None

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=8,
    train=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=train_anno,
        ann_file=train_anno,
        nsweeps=nsweeps,
        class_names=class_names,
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=val_anno,
        test_mode=True,
        ann_file=val_anno,
        nsweeps=nsweeps,
        class_names=class_names,
        version='v1.0-trainval',
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=test_anno,
        ann_file=test_anno,
        nsweeps=nsweeps,
        class_names=class_names,
        pipeline=test_pipeline,
    ),
)


optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# optimizer
optimizer = dict(
    type="adam", amsgrad=0.0, wd=0.01, fixed_wd=True, moving_average=False,
)
lr_config = dict(
    type="one_cycle", lr_max=0.0001, moms=[0.95, 0.85], div_factor=10.0, pct_start=0.4,
)

checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=100,
    hooks=[
        dict(type="TextLoggerHook"),
        # dict(type='TensorboardLoggerHook')
    ],
)
# yapf:enable
# runtime settings
total_epochs = 40
device_ids = range(2)
dist_params = dict(backend="nccl", init_method="env://")
log_level = "INFO"
work_dir = './work_dirs/{}/'.format(__file__[__file__.rfind('/') + 1:-3])
load_from = "None"
resume_from = None
workflow = [('train', 1), ('val', 1)]
gpus = 1
LOCAL_RANK = 0