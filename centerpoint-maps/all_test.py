import argparse
import copy
import logging
import os

import torch
from det3d import __version__, torchie
from det3d.datasets import build_dataloader, build_dataset
from det3d.models import build_detector
from det3d.torchie import Config
from det3d.torchie.apis import (
    batch_processor,
    build_optimizer,
    get_root_logger,
    init_dist,
    set_random_seed,
    train_detector,
)
from det3d.torchie.trainer import get_dist_info, load_checkpoint
from det3d.torchie.trainer.utils import all_gather, synchronize
from torch.nn.parallel import DistributedDataParallel
import pickle
import time

config = "./configs/centerpoint/test_generate.py"
work_dir = "./work_dirs/from_0_car_test/"
generate_3D_proposals = True


def parse_args():
    parser = argparse.ArgumentParser(description="Train a detector")
    parser.add_argument(
        "--txt_result",
        type=bool,
        default=False,
        help="whether to save results to standard KITTI format of txt type",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="number of gpus to use " "(only applicable to non-distributed training)",
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--speed_test", action="store_true")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--testset", action="store_true")

    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    return args


logname = os.path.join(work_dir, "eval_result.log")
logging.basicConfig(filename=logname,
                    filemode='a',
                    format='%(asctime)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)


def save_pred(pred, root):
    with open(os.path.join(root, "prediction.pkl"), "wb") as f:
        pickle.dump(pred, f)


def test(config, checkpoint, work_dir, speed_test=True):
    logging.info(f"Running eval on {checkpoint}")
    args = parse_args()
    # torch.manual_seed(0)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # np.random.seed(0)

    cfg = Config.fromfile(config)
    cfg.local_rank = args.local_rank

    # update configs according to CLI args
    if work_dir is not None:
        cfg.work_dir = work_dir

    distributed = False
    if "WORLD_SIZE" in os.environ:
        distributed = int(os.environ["WORLD_SIZE"]) > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")

        cfg.gpus = torch.distributed.get_world_size()
    else:
        cfg.gpus = args.gpus

    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)

    if args.testset:
        print("Use Test Set")
        dataset = build_dataset(cfg.data.test)
    else:
        print("Use Val Set")
        dataset = build_dataset(cfg.data.val)

    data_loader = build_dataloader(
        dataset,
        batch_size=cfg.data.samples_per_gpu if not speed_test else 1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
    )

    checkpoint = load_checkpoint(model, checkpoint, map_location="cpu")

    # put model on gpus
    model = model.cuda()

    model.eval()
    mode = "val"

    if cfg.local_rank == 0:
        prog_bar = torchie.ProgressBar(len(data_loader.dataset) // cfg.gpus)

    detections = {}
    cpu_device = torch.device("cpu")

    start = time.time()

    start = int(len(dataset) / 3)
    end = int(len(dataset) * 2 / 3)

    time_start = 0
    time_end = 0

    for i, data_batch in enumerate(data_loader):
        if i == start:
            torch.cuda.synchronize()
            time_start = time.time()

        if i == end:
            torch.cuda.synchronize()
            time_end = time.time()

        with torch.no_grad():
            outputs = batch_processor(
                model, data_batch, train_mode=False, local_rank=args.local_rank,
            )
        for output in outputs:
            token = output["metadata"]["token"]
            for k, v in output.items():
                if k not in [
                    "metadata",
                ]:
                    output[k] = v.to(cpu_device)
            detections.update(
                {token: output, }
            )
            if generate_3D_proposals:
                os.makedirs('output_3d', exist_ok=True)
                file_name = "./output_3d/" + str(token) + '.pt'
                torch.save(output, file_name)

            if args.local_rank == 0:
                prog_bar.update()

    synchronize()

    all_predictions = all_gather(detections)
    process_time = (time_end - time_start) / (end - start)
    print("\n Total time per frame: {}".format(process_time))
    logging.info("\n Total time per frame: {}".format(process_time))
    if args.local_rank != 0:
        return

    predictions = {}
    for p in all_predictions:
        predictions.update(p)

    if not os.path.exists(work_dir):
        os.makedirs(work_dir)

    save_pred(predictions, work_dir)
    with open(os.path.join(work_dir, 'prediction.pkl'), 'rb') as f:
        predictions = pickle.load(f)

    result_dict, _ = dataset.evaluation(copy.deepcopy(predictions), output_dir=work_dir, testset=args.testset)
    logging.info("\n")
    if result_dict is not None:
        for k, v in result_dict["results"].items():
            print(f"Evaluation {k}: {v}")
            logging.info(f"Evaluation {k}: {v}")

    if args.txt_result:
        assert False, "No longer support kitti"


if __name__ == "__main__":

    for file in os.listdir(work_dir):
        if (file.endswith(".pth")) & (file != "latest.pth"):
            try:
                checkpoint = os.path.join(work_dir, file)
                test(config, checkpoint, work_dir)
            except:
                pass


