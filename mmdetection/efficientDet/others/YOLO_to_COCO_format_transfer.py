import os
import json
import time

from PIL import Image


# Read the label files (.txt) to extarct bounding box information and store in COCO format

path_labels_train = os.fsencode("/home/foj-sy/ipa_one_stage_detection/ipa_one_stage_detection/yolov4/model_data/train.txt")
path_labels_val = os.fsencode("/home/foj-sy/ipa_one_stage_detection/ipa_one_stage_detection/yolov4/model_data/val.txt")

coco_format_save_path_train = '/home/foj-sy/Documents/datasets/Doors/images/annotations/instances_train.json'
coco_format_save_path_val = '/home/foj-sy/Documents/datasets/Doors/images/annotations/instances_val.json'

# Category file, one category per line
yolo_format_classes_path = '/home/foj-sy/ipa_one_stage_detection/ipa_one_stage_detection/yolov4/model_data/class.txt'


def transfer_yolo_to_coco(path_labels, yolo_format_classes_path, coco_format_save_path):
    # Read the categories file and extract all categories
    with open(yolo_format_classes_path, 'r') as f1:
        lines1 = f1.readlines()
    categories = []
    for j, label in enumerate(lines1):
        label = label.strip()
        categories.append({'id': j + 1, 'name': label, 'supercategory': label})

    write_json_context = dict()
    write_json_context['info'] = {'description': '', 'url': '', 'version': '', 'year': 2022, 'contributor': '',
                                  'date_created': '2022-04-12 11:00:08.5'}
    write_json_context['licenses'] = [{'id': 1, 'name': None, 'url': None}]
    write_json_context['categories'] = categories
    write_json_context['images'] = []
    write_json_context['annotations'] = []

    # Read the YOLO formatted label files (.txt) to extarct bounding box information and store in COCO format
    with open(path_labels, 'r') as f2:
        lines2 = f2.readlines()

    file_number = 1
    num_bboxes = 1

    for i, line in enumerate(lines2):  # for loop runs for number of annotations labelled in an image
        line = line.split(' ')
        img_path = line[0].strip()
        img_context = {}

        img_name = os.path.basename(img_path)  # name of the file without the extension
        # print(img_name)

        im = Image.open(img_path)
        width, height = im.size
        # height, width = cv2.imread(img_path).shape[:2]
        img_context['file_name'] = img_path
        img_context['height'] = height
        img_context['width'] = width
        img_context['date_captured'] = '2022-04-12 11:00:08.5'
        img_context['id'] = file_number  # image id
        img_context['license'] = 1
        img_context['coco_url'] = ''
        img_context['flickr_url'] = ''
        write_json_context['images'].append(img_context)

        bboxs = []
        for element in line[1:]:
            bboxs.append(element.strip())

        for bbox in bboxs:
            bbox_dict = {}
            bbox = bbox.split(',')
            x1, y1, x2, y2, class_id = bbox[0:]
            x1, y1, x2, y2, class_id = float(x1), float(y1), float(x2), float(y2), int(class_id)

            bbox_dict['id'] = num_bboxes
            bbox_dict['image_id'] = file_number
            bbox_dict['category_id'] = class_id + 1
            bbox_dict['iscrowd'] = 0  # There is an explanation before
            h, w = abs(y2 - y1), abs(x2 - x1)
            bbox_dict['area'] = h * w
            x_coco = round(x1)
            y_coco = round(y1)
            if x_coco < 0:  # check if x_coco extends out of the image boundaries
                x_coco = 1
            if y_coco < 0:  # check if y_coco extends out of the image boundaries
                y_coco = 1
            bbox_dict['bbox'] = [x_coco, y_coco, w, h]
            bbox_dict['segmentation'] = [
                [x_coco, y_coco, x_coco + w, y_coco, x_coco + w, y_coco + h, x_coco, y_coco + h]]
            write_json_context['annotations'].append(bbox_dict)
            num_bboxes += 1

        file_number = file_number + 1
        continue

    # Finally done, save!
    with open(coco_format_save_path, 'w') as fw:
        json.dump(write_json_context, fw)


if __name__ == '__main__':
    print("###### start to process train labels ######")
    transfer_yolo_to_coco(path_labels_train, yolo_format_classes_path, coco_format_save_path_train)
    print("###### train labels completed! ######")
    time.sleep(1)
    print("###### start to process val labels ######")
    transfer_yolo_to_coco(path_labels_val, yolo_format_classes_path, coco_format_save_path_val)
    print("###### val labels completed! ######")