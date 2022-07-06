##2D Label Parser
Partly modified from [2D label parser for Yolov3 darknet](https://github.com/AlizadehAli/2D_label_parser).

The contents are listed below, for concrete useage, please refer to project readme.

```   
centerpoint_maps     
└── 2D_label_parser
       ├── labels                   <-- json labels (from export_2D_anno_as_json.py)
       ├── target_labels            <-- output txt label file
                ├── CAM_BACK        <-- emtpy
                ├── CAM_BACK_RIGHT  <-- emtpy
                ├── CAM_BACK_LEFT   <-- emtpy
                └── ...
       ├── label_parser.py          <-- transfer json label to YOLO format txt file
       ├── show_anno_coco.py        <-- check coco format annotation
       ├── show_pt.py               <-- show .pt file
       ├── export_2D_anno_as_json.py <-- trasfer nuScenes 3D anno to 2D
       ├── check_anno_number.py     <-- check anno number from COCO format anno
       └── transfer_txt_2_COCO.py
```

##Original ReadMe

2D label parser for Yolov3 darknet
---
This repository contains the parser functions to convert COCO/YOLO, BDD, and nuscenes json format to txt format needed by darknet Yolov3.

2D bounding box format is taken as [xmin, xmax, ymin, ymax]. Moreover, the object classes from nuscenes dataset is reduced from 23 to 10 regarding our needs; "pedestrian, bicycle, motorcycle, car, bus, truck, emergency, construction, movable object, and bicycle_rack" are the objects of interest in this parser. You can eaily customize it according to your needs in the nuscenes_parser function.

