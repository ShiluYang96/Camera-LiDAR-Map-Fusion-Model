"""
process KITTI dataset for 2D detector (in process)
"""

import os
from tqdm import tqdm
from pathlib import Path


label_dir = r"data/kitti/training/label_2"
out_dir_easy = "data/kitti/training/label_2_easy"
out_dir_mode = "data/kitti/training/label_2_mode"
out_dir_hard = "data/kitti/training/label_2_hard"

# read all gt files
gts = os.listdir(label_dir)

# create output dir
Path(out_dir_easy).mkdir(parents=True, exist_ok=True)
Path(out_dir_mode).mkdir(parents=True, exist_ok=True)
Path(out_dir_hard).mkdir(parents=True, exist_ok=True)

with tqdm(total=len(gts)) as pbar:
    for gt in gts:
        gt_path = os.path.join(label_dir, gt)
        frame_id = gt.split('.')[0]
        out_file_easy = os.path.join(out_dir_easy, (frame_id + '.txt'))
        out_file_mode = os.path.join(out_dir_mode, (frame_id + '.txt'))
        out_file_hard = os.path.join(out_dir_hard, (frame_id + '.txt'))

        context_easy = []
        context_mode = []
        context_hard = []

        with open(gt_path) as f:
            lines = f.readlines()

        for line in lines:
            line_check = line.split(" ")
            height = float(line_check[7]) - float(line_check[5])
            truncation = float(line_check[1])
            occlusion_level = int(line_check[2])

            if (height >= 40) & (occlusion_level == 0) & (truncation <= 0.15):
                context_easy.append(line)
            elif (height >= 25) & (occlusion_level <= 1) & (truncation <= 0.3):
                context_mode.append(line)
            elif (height >= 25) & (occlusion_level <= 2) & (truncation <= 0.50):
                context_hard.append(line)

        # write
        with open(out_file_easy, 'w+') as f:
            for line in context_easy:
                f.write(line)
        with open(out_file_mode, 'w+') as f:
            for line in context_mode:
                f.write(line)
        with open(out_file_hard, 'w+') as f:
            for line in context_hard:
                f.write(line)
        pbar.update(1)