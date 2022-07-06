from torch.utils.data import DataLoader
from late_fusion.utils.robotino_dataset import clocs_data
import argparse
import torch
from late_fusion.utils import fusion,nms
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.utils import common_utils
from tqdm import tqdm
from late_fusion.utils.Focaloss import SigmoidFocalClassificationLoss
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description='Train network')

    parser.add_argument('--cfg_file', type=str, default='../tools/cfgs/robotino_models/pointpillar.yaml',
                        help='specify the config for training')
    parser.add_argument('--rootpath', type=str, default='../data',
                        help='data root path')
    parser.add_argument('--d2path', type=str, default='../data/2D_proposals',
                        help='2d prediction path')
    parser.add_argument('--d3path', type=str, default='../data/3D_proposals',
                        help='3d prediction path')
    parser.add_argument('--epochs', type=int, default=12,
                        help='training epochs')
    parser.add_argument('--log-path', type=str, default='../log/late_fusion',
                        help='log path')
    parser.add_argument('--generate', type=bool, default=False,
                        help='whether generate input data')
    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])
    return args, cfg


def train(fusion_layer, train_data, optimizer, epoch, logf):
    cls_loss_sum = 0
    optimizer.zero_grad()
    step = 1
    display_step = 500
    for fusion_input,tensor_index,positives,negatives,one_hot_targets,label_n,idx in tqdm(train_data):
        fusion_input = fusion_input.cuda()
        tensor_index = tensor_index.reshape(-1,2)
        tensor_index = tensor_index.cuda()
        positives = positives.cuda()
        negatives = negatives.cuda()
        one_hot_targets = one_hot_targets.cuda()
        cls_preds,flag = fusion_layer(fusion_input,tensor_index)

        negative_cls_weights = negatives.type(torch.float32) * 1.0
        cls_weights = negative_cls_weights + 1.0 * positives.type(torch.float32)
        pos_normalizer = positives.sum(1, keepdim=True).type(torch.float32)
        cls_weights /= torch.clamp(pos_normalizer, min=1.0)
        if flag==1:
            cls_preds = cls_preds[:,:one_hot_targets.shape[1],:]
            cls_losses = Focal._compute_loss(cls_preds, one_hot_targets, cls_weights.cuda())  # [N, M]
            cls_losses_reduced = cls_losses.sum()/(label_n.item()+1)
            # cls_losses_reduced = cls_losses.sum()
            cls_loss_sum = cls_loss_sum + cls_losses.sum()
            cls_losses_reduced.backward()
            optimizer.step()
            optimizer.zero_grad()
        step = step + 1
        if step%display_step == 0:
            print("epoch:",epoch, " step:", step, " and the cls_loss is :",cls_loss_sum.item()/display_step, file=logf)
            print("epoch:",epoch, " step:", step, " and the cls_loss is :",cls_loss_sum.item()/display_step)
            cls_loss_sum = 0


def eval(net, val_data, logf, log_path, epoch, cfg, eval_set, logger):
    net.eval()
    det_annos = []
    
    logger.info("#################################")
    print("#################################", file=logf)
    logger.info("# EVAL" + str(epoch))
    print("# EVAL"+ str(epoch), file=logf)
    logger.info("#################################")
    print("#################################", file=logf)
    logger.info("Generate output labels...")
    print("Generate output labels...", file=logf)
    for fusion_input,tensor_index,path in tqdm(val_data):
        fusion_input = fusion_input.cuda()
        tensor_index = tensor_index.reshape(-1,2)
        tensor_index = tensor_index.cuda()
        _3d_result = torch.load(path[0])[0]
        fusion_cls_preds,flag = net(fusion_input,tensor_index)
        cls_preds = fusion_cls_preds.reshape(-1).cpu()
        cls_preds = torch.sigmoid(cls_preds)
        cls_preds = cls_preds[:len(_3d_result['score'])]
        _3d_result['score'] = cls_preds.detach().cpu().numpy()
        box_preds = torch.tensor(_3d_result['boxes_lidar']).cuda()
        selected = nms.nms(cls_preds, box_preds, cfg.MODEL.POST_PROCESSING)
        selected = selected.numpy()
        for key in _3d_result.keys():
            if key == 'frame_id':
                continue
            _3d_result[key] = _3d_result[key][selected]
        det_annos.append(_3d_result)
    
    logger.info("Generate output done")
    print("Generate output done", file=logf)
    torch.save(det_annos,log_path+'/'+'det_result'+'/'+str(epoch)+'.pt')
    result_str, result_dict = eval_set.evaluation(
        det_annos, cfg.CLASS_NAMES,
        eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC
    )
    print(result_str, file=logf)
    logger.info(result_str)
    net.train()


if __name__ == "__main__":
    Focal = SigmoidFocalClassificationLoss()

    args, cfg = parse_args()
    root_path = args.rootpath
    _2d_path = args.d2path
    _3d_path = args.d3path

    input_data = root_path + '/input_tensor'
    log_path = args.log_path
    log_Path = Path(log_path)
    if not log_Path.exists():
        log_Path.mkdir(parents=True)
    result_Path = log_Path / 'det_result'
    if not result_Path.exists():
        result_Path.mkdir(parents=True)
    logf = open(log_path+'/log.txt', 'a')
    log_file = log_Path / 'log.txt'
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    if args.generate :
        train_dataset = clocs_data(_2d_path, _3d_path, input_data, root_path)
        train_dataset.generate_input()

    train_dataset = clocs_data(_2d_path, _3d_path, input_data, root_path)
    val_dataset = clocs_data(_2d_path, _3d_path, input_data, root_path, val=True)

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

    eval_set, _, __ = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=1,
        root_path=Path(root_path + '/robotino'),
        dist=False, workers=8, logger=logger, training=False
    )

    fusion_layer = fusion.fusion()
    fusion_layer.cuda()

    optimizer = torch.optim.Adam(fusion_layer.parameters(), lr=3e-3, betas=(0.9, 0.99),weight_decay=0.01)

    for epoch in range(args.epochs):
        train(fusion_layer, train_data, optimizer, epoch, logf)
        torch.save(fusion_layer, log_path+'/'+str(epoch)+'.pt')
        eval(fusion_layer, val_data, logf, log_path, epoch, cfg, eval_set, logger)

