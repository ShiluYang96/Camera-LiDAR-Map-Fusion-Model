from mmdet.apis import init_detector, inference_detector
import os
import numpy as np
from tqdm import tqdm
from pathlib import Path
# python tools/train.py configs/yolox/yolox_l_8x8_300e_coco.py --gpu-id=1
# python tools/train.py configs/cascade_rcnn/cascade_rcnn_r50_fpn_1x_nuScenes.py --gpu-id=1
# python tools/train.py configs/yolox/yolox_s_8x8_300e_nuscenes.py --gpus=1
# python tools/train.py configs/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco.py --gpus=1
# python tools/train.py configs/yolox/yolox_s_8x8_300e_coco.py --gpus=1
# python tools/test.py configs/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco.py work_dirs/cascade_rcnn_r50_fpn_1x_coco/epoch_4.pth --eval mAP
# python tools/test.py configs/cascade_rcnn/cascade_rcnn_r50_fpn_1x_nuScenes.py work_dirs/cascade_rcnn_r50_fpn_1x_nuScenes/epoch_5.pth --eval bbox --gpu-id=1
# python tools/test.py configs/yolox/yolox_s_8x8_300e_coco.py /home/yang/mmdetection/work_dirs/yolox_s_8x8_300e_coco/epoch_20.pth --eval mAP


# give the configuration file and checkpoint file
config_file = "/home/yang/mmdetection/work_dirs/yolox_s_8x8_300e_nuscenes/yolox_s_8x8_300e_nuscenes.py"
checkpoint_file = r"/home/yang/mmdetection/work_dirs/yolox_s_8x8_300e_nuscenes/best_bbox_mAP_epoch_27.pth"
img_dir = r"/home/yang/centerpoint_maps/data/nuScenes/samples"
out_dir = "Output_2D"
Data_type = "nuScenes"  # or KITTI
Proposal_num_per_image = 100

def predict(model, img):
    result = inference_detector(model, img)
    bboxes_scores = np.vstack(result)
    bboxes_raw = bboxes_scores[:, :4]

    score = bboxes_scores[:, 4]
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(result)
    ]
    labels = np.concatenate(labels)
    return labels, bboxes_raw, score


def generate_2D_proposals(model, img_dir, out_dir):
    # create output dir
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # read all images
    print("********* Start generating proposals for  all images in ", img_dir, " *********")
    imgs = os.listdir(img_dir)

    with tqdm(total=len(imgs)) as pbar:
        for img in imgs:
            img_path = os.path.join(img_dir, img)
            # get image name
            frame_id = img.split('.')[0]
            out_file = os.path.join(out_dir, (frame_id + '.txt'))
            # prediction
            labels, bboxes, score = predict(model, img_path)

            context = []
            for i in range(len(labels)):
                if (labels[i] == 0) & (len(context) < 200):  # save 100 highest proposals
                    bbox = bboxes[i]
                    if ((i + 1) < len(labels)) & (len(context) < (Proposal_num_per_image-1)):
                        write_line = "Car -1 -1 -10 {:.2f} {:.2f} {:.2f} {:.2f} -1 -1 -1 -1000 -1000 -1000 -10 {:.4f} \n". \
                            format(bbox[0], bbox[1], bbox[2], bbox[3], score[i])
                    else:  # if the last line
                        write_line = "Car -1 -1 -10 {:.2f} {:.2f} {:.2f} {:.2f} -1 -1 -1 -1000 -1000 -1000 -10 {:.4f} ". \
                            format(bbox[0], bbox[1], bbox[2], bbox[3], score[i])

                    context.append(write_line)
            # write all 2d proposals
            with open(out_file, 'w+') as f:
                for line in context:
                    f.write(line)
            pbar.update(1)


if __name__ == "__main__":
    # build model
    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    if Data_type == "nuScenes":
        channel_list = ['CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT', 'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT']
        for ref_chan in channel_list:
            sub_img_dir = os.path.join(img_dir, ref_chan)
            sub_out_dir = os.path.join(out_dir, ref_chan)
            generate_2D_proposals(model, sub_img_dir, sub_out_dir)
    else:
        generate_2D_proposals(model, img_dir, out_dir)
