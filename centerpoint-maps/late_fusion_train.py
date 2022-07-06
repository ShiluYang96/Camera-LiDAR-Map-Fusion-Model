import copy
import os
import pickle
import argparse

import torch
from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader

from nuscenes import NuScenes
from det3d.torchie.trainer.utils import synchronize, all_gather
from det3d.datasets import build_dataset
from det3d.torchie import Config
from det3d.models.bbox_heads.mg_head import _circle_nms

from late_fusion.late_fusion_dataset import Late_fusion_dataset
from late_fusion import common_utils, fusion
from late_fusion.Focaloss import SigmoidFocalClassificationLoss


def parse_args():
    parser = argparse.ArgumentParser(description='Train network')

    parser.add_argument('--cfg_file', type=str, default="configs/centerpoint/nusc_late_fusion.py",
                        help='specify the config for training')
    parser.add_argument('--version', type=str, default="v1.0-trainval",
                        help='nuScenes version')
    parser.add_argument('--rootpath', type=str, default='data/nuScenes',
                        help='data root path')
    parser.add_argument('--d2path', type=str, default='data/Output_2D_YOLO_less',
                        help='2d prediction path')
    parser.add_argument('--d3path', type=str, default='data/output_3d',
                        help='3d prediction path')
    parser.add_argument('--epochs', type=int, default=30,
                        help='training epochs')
    parser.add_argument('--log_path', type=str, default='./log/late_fusion',
                        help='log path')
    parser.add_argument('--input_path', type=str, default='input_tensor',
                        help='name of input tensor dir')
    parser.add_argument('--generate', type=bool, default=False,
                        help='whether generate input data')
    args = parser.parse_args()

    cfg = Config.fromfile(args.cfg_file)
    return args, cfg


def save_pred(pred, root):
    with open(os.path.join(root, "prediction.pkl"), "wb") as f:
        pickle.dump(pred, f)


def train(fusion_layer, train_data, optimizer, epoch, logf):
    cls_loss_sum = 0
    optimizer.zero_grad()
    step = 1
    display_step = 500
    print(len(train_data))
    for fusion_input, tensor_index, positives, negatives, one_hot_targets, label_n, idx in tqdm(train_data):
        fusion_input = fusion_input.cuda()
        tensor_index = tensor_index.reshape(-1, 2)
        tensor_index = tensor_index.cuda()
        positives = positives.cuda()
        negatives = negatives.cuda()
        one_hot_targets = one_hot_targets.cuda()
        cls_preds, flag = fusion_layer(fusion_input, tensor_index)

        negative_cls_weights = negatives.type(torch.float32) * 1.0
        cls_weights = negative_cls_weights + 1.0 * positives.type(torch.float32)
        pos_normalizer = positives.sum(1, keepdim=True).type(torch.float32)
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)
        if flag == 1:
            cls_preds = cls_preds[:, :one_hot_targets.shape[1], :]
            cls_losses = Focal._compute_loss(cls_preds, one_hot_targets, cls_weights.cuda())  # [N, M]
            cls_losses_reduced = cls_losses.sum() / (label_n.item() + 1)
            cls_loss_sum = cls_loss_sum + cls_losses.sum()
            cls_losses_reduced.backward()
            optimizer.step()
            optimizer.zero_grad()
        step = step + 1
        if step % display_step == 0:
            print("epoch:", epoch, " step:", step, " and the cls_loss is :", cls_loss_sum.item() / display_step,
                  file=logf)
            print("epoch:", epoch, " step:", step, " and the cls_loss is :", cls_loss_sum.item() / display_step)
            cls_loss_sum = 0


def eval(net, val_data, logf, log_path, epoch, eval_set, logger, cfg):
    net.eval()

    logger.info("#################################")
    print("#################################", file=logf)
    logger.info("# EVAL" + str(epoch))
    print("# EVAL" + str(epoch), file=logf)
    logger.info("#################################")
    print("#################################", file=logf)
    logger.info("Generate output labels...")
    print("Generate output labels...", file=logf)
    detections = {}

    for fusion_input, tensor_index, path in tqdm(val_data):
        fusion_input = fusion_input.cuda()
        tensor_index = tensor_index.reshape(-1, 2)
        tensor_index = tensor_index.cuda()
        _3d_result = torch.load(path[0])
        fusion_cls_preds, flag = net(fusion_input, tensor_index)
        cls_preds = fusion_cls_preds.reshape(-1).cpu()
        cls_preds = torch.sigmoid(cls_preds)
        cls_preds = cls_preds[:len(_3d_result['scores'])]
        # cls_preds  = cls_preds * 2
        # filter out the proposals under score threshold
        thresh = torch.tensor(
            [cfg.test_cfg.score_threshold], device=cls_preds.device
        ).type_as(cls_preds)
        top_scores_keep = cls_preds >= thresh
        cls_preds = cls_preds[top_scores_keep]

        # apply nms
        _3d_result['scores'] = cls_preds.detach()
        _3d_result["box3d_lidar"] = _3d_result["box3d_lidar"][top_scores_keep]
        _3d_result["label_preds"] = _3d_result["label_preds"][top_scores_keep]
        centers = _3d_result["box3d_lidar"][:, [0, 1]].detach()
        boxes = torch.cat([centers, _3d_result['scores'].view(-1, 1)], dim=1)
        selected = _circle_nms(boxes, min_radius=4, post_max_size=83)

        for key in _3d_result.keys():
            if key == 'metadata':
                continue
            _3d_result[key] = _3d_result[key][selected]

        token = _3d_result["metadata"]["token"]
        for k, v in _3d_result.items():
            if k not in [
                "metadata",
            ]:
                _3d_result[k] = v
        detections.update(
            {token: _3d_result, }
        )

    synchronize()
    all_predictions = all_gather(detections)

    predictions = {}
    for p in all_predictions:
        predictions.update(p)

    if not os.path.exists(log_path):
        os.makedirs(log_path)

    save_pred(predictions, log_path)
    with open(os.path.join(log_path, 'prediction.pkl'), 'rb') as f:
        predictions = pickle.load(f)

    result_dict, _ = eval_set.evaluation(copy.deepcopy(predictions), output_dir=log_path)

    logger.info("\n")
    if result_dict is not None:
        for k, v in result_dict["results"].items():
            print(f"Evaluation {k}: {v}")
            logger.info(f"Evaluation {k}: {v}")

    net.train()


if __name__ == "__main__":
    Focal = SigmoidFocalClassificationLoss()

    args, cfg = parse_args()
    root_path = args.rootpath
    version = args.version
    _2d_path = args.d2path
    _3d_path = args.d3path
    input_path = args.input_path
    nusc = NuScenes(version=version, dataroot=root_path, verbose=True)

    input_data = root_path + '/' + input_path
    train_ind_path = root_path + '/index/train.txt'
    val_ind_path = root_path + '/index/val.txt'
    info_path_train = root_path + '/infos_train_10sweeps_withvelo_filter_True.pkl'
    info_path_val = root_path + '/infos_val_10sweeps_withvelo_filter_True.pkl'

    log_path = args.log_path
    log_Path = Path(log_path)
    if not log_Path.exists():
        log_Path.mkdir(parents=True)
    logf = open(log_path + '/log.txt', 'a')
    log_file = log_Path / 'log.txt'
    logger = common_utils.create_logger(log_file, rank=0)

    train_dataset = Late_fusion_dataset(nusc, root_path, _2d_path, _3d_path, train_ind_path, input_data, info_path_train, info_path_val)
    if args.generate:
        train_dataset.generate_input()
    val_dataset = Late_fusion_dataset(nusc, root_path, _2d_path, _3d_path, val_ind_path, input_data, info_path_train, info_path_val, val=True)

    train_data = DataLoader(
        train_dataset,
        batch_size=1,
        num_workers=8,
        pin_memory=True
    )

    val_data = DataLoader(
        val_dataset,
        batch_size=1,
        num_workers=8,
        pin_memory=True
    )

    eval_dataset = build_dataset(cfg.data.val)

    fusion_layer = fusion.fusion()
    fusion_layer.cuda()
    optimizer = torch.optim.Adam(fusion_layer.parameters(), lr=3e-4, betas=(0.9, 0.99), weight_decay=0.01)

    for epoch in range(args.epochs):
        train(fusion_layer, train_data, optimizer, epoch, logf)
        torch.save(fusion_layer.state_dict(), log_path + '/' + str(epoch) + '.pth')
        # torch.save(fusion_layer, log_path + '/' + str(epoch) + '.pt')
        eval(fusion_layer, val_data, logf, log_path, epoch, eval_dataset, logger, cfg)
