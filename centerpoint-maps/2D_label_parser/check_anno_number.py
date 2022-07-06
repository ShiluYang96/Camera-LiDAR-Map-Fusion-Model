import os.path

anno_dir = "/home/yang/centerpoint_maps/2D_label_parser/target_labels"
channel_list = ['CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT', 'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT']

anno_dict = {

}
for channel in channel_list:
    anno_dict[channel] = []
    anno_paths = os.path.join(anno_dir, channel)
    print(len(os.listdir(anno_paths)))
    for file in os.listdir(anno_paths):
        anno_file = os.path.join(anno_paths, file)
        with open(anno_file, 'r') as f:
            lines = f.readlines()
        for line in lines:
            obj = line.split(' ')[0]
            if obj == "0":
                anno_dict[channel].append(obj)

    print(len(anno_dict[channel]))



