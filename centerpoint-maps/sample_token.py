import json
from nuscenes import NuScenes
from pathlib import Path

sample_file = "data/nuScenes/v1.0-trainval/sample.json"
sample_data_file = "data/nuScenes/v1.0-trainval/sample_data.json"
sample_annotation_file = "data/nuScenes/v1.0-trainval/sample_annotation.json"
scene_file = "data/nuScenes/v1.0-trainval/scene.json"
root_path = "data/nuScenes"
version = "v1.0-trainval"


def _get_available_scenes(nusc):
  available_scenes = []
  print("total scene num:", len(nusc.scene))
  for scene in nusc.scene:
    scene_token = scene["token"]
    scene_rec = nusc.get("scene", scene_token)
    sample_rec = nusc.get("sample", scene_rec["first_sample_token"])
    sd_rec = nusc.get("sample_data", sample_rec["data"]["LIDAR_TOP"])
    has_more_frames = True
    scene_not_exist = False
    while has_more_frames:
      lidar_path, boxes, _ = nusc.get_sample_data(sd_rec["token"])
      if not Path(lidar_path).exists():
        scene_not_exist = True
        break
      else:
        break
      if not sd_rec["next"] == "":
        sd_rec = nusc.get("sample_data", sd_rec["next"])
      else:
        has_more_frames = False
    if scene_not_exist:
      continue
    available_scenes.append(scene)
  print("exist scene num:", len(available_scenes))
  return available_scenes


nusc = NuScenes(version=version, dataroot=root_path, verbose=True)
available_scenes = _get_available_scenes(nusc)
# available_scene_names = [s["name"] for s in available_scenes]
available_scene_token = [s["token"] for s in available_scenes]


with open(scene_file, 'r') as f:
  scenes = json.load(f)
out_scenes = []
with open('scene.json', 'w') as f:
  for scene in scenes:
      scene_token = scene['token']
      if scene_token in available_scene_token:
        out_scenes.append(scene)
  json.dump(out_scenes, f)


with open(sample_file, 'r') as f:
  samples = json.load(f)

out_sample = []
with open('sample.json', 'w') as f:
  for sample in samples:
      sample_scene_token = sample['scene_token']
      if sample_scene_token in available_scene_token:
        out_sample.append(sample)
  json.dump(out_sample, f)

available_sample_token = [s["token"] for s in out_sample]

with open(sample_data_file, 'r') as f:
  sample_datas = json.load(f)
out_sample_data = []
with open('sample_data.json', 'w') as f2:
  for sample_data in sample_datas:
      sample_token = sample_data['sample_token']
      if sample_token in available_sample_token:
        out_sample_data.append(sample_data)
  json.dump(out_sample_data, f2)

with open(sample_annotation_file, 'r') as f:
  sample_annotations = json.load(f)
out_sample_annotation = []
with open('sample_annotation.json', 'w') as f3:
  for sample_annotation in sample_annotations:
      sample_token = sample_annotation['sample_token']
      if sample_token in available_sample_token:
        out_sample_annotation.append(sample_annotation)
  json.dump(out_sample_annotation, f3)