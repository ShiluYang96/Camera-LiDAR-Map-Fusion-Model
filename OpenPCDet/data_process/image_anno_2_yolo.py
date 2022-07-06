import argparse
import json

import yaml
import os
import cv2

Class_name = ['robotino']

def show_bbox(img_full_name, left, right, top, bottom):
    # load the image and scale it
    image = cv2.imread(img_full_name)
    thick = 3
    color = (0, 0, 255)
    cv2.namedWindow("image window", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("image window", 1200, 1200)

    cv2.rectangle(image, (left, top), (right, bottom), color, thick)

    cv2.imshow('image window', image)
    # add wait key. window waits until user presses a key
    cv2.waitKey(1000)
    # and finally destroy/close all open windows
    cv2.destroyAllWindows()


def yolo_line_to_shape(x_center, y_center, w, h):

        x_min = int(float(x_center) - float(w) / 2)
        x_max = int(float(x_center) + float(w) / 2)
        y_min = int(float(y_center) - float(h) / 2)
        y_max = int(float(y_center) + float(h) / 2)

        return x_min, y_min, x_max, y_max


# transfer label tool output to train.txt or val.txt
def label_tool_2_txt(LabelImg_gt_dir, output_path, images_dir):
    with open(output_path, 'w') as f:
        for json_name in os.listdir(LabelImg_gt_dir):

            # read json file
            with open(os.path.join(LabelImg_gt_dir, json_name)) as j_f:
                json_data = yaml.load(j_f, Loader=yaml.FullLoader)[0]
                j_f.close()

            # write img path
            image_name = json_data["image"]
            full_image_name = os.path.join(images_dir, image_name)
            f.write(full_image_name)

            for bndBox in json_data["annotations"]:
                type = bndBox["label"]
                type = Class_name.index(type)
                x_center = bndBox["coordinates"]["x"]
                y_center = bndBox["coordinates"]["y"]
                w = bndBox["coordinates"]["width"]
                h = bndBox["coordinates"]["height"]
                left, bottom, right, top = yolo_line_to_shape(x_center, y_center, w, h)
                box_info = " %d,%d,%d,%d,%d" % (left, bottom, right, top, int(type))
                if display_bbox:
                    show_bbox(full_image_name, left, right, top, bottom)
                f.write(box_info)  # write bbox info
            f.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", help="Path of dataset root", type=str, default='/home/foj-sy/Downloads/dataset/Robotino')
    parser.add_argument("--display_bbox", help="display bounding box during processing", type=bool, default=False)
    parser.add_argument("--target_dir", help="annotaion dir, val or train", type=str, default="train")
    args = parser.parse_args()
    dataset_dir = args.dataset_dir
    display_bbox = args.display_bbox
    target_dir = args.target_dir

    images_dir = os.path.join(dataset_dir, 'images', target_dir)
    output_path = 'yolo_' + target_dir + '.txt'
    LabelImg_gt_dir = os.path.join(dataset_dir, "annos/" + target_dir + '/')

    label_tool_2_txt(LabelImg_gt_dir, output_path, images_dir)
