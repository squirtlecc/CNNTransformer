import argparse
import os
# from torchsummary import summary
from utils.config import Config
from app.mediator import Mediator
import torch


def main() -> None:
    """Main Function
    """
    args = parseArgs()


    gpu = ','.join(str(gpu) for gpu in args.gpus)

    # CUDA_VISIBLE_DEVICES MUST SETTING BEFORE IMPORT TORCH
    # os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    device = "cuda:{}".format(gpu) if args.gpus[0] > -1 else "cpu"
    torch.cuda.set_device(device)
    # torch.cuda.set_device(args.local_rank)


    cfg = Config.fromfile(args.config)
    cfg = updateCfgs(cfg, args)

    mediator = Mediator(cfg)

    if args.validate:
        mediator.validate()
    else:
        mediator.train()


def updateCfgs(cfg, args) -> Config:


    if args.validate:
        cfg.logs_dir = args.work_dir
    cfg.dataset.work_dir = args.work_dir
    cfg.random_seed = args.random_seed
    cfg.load_from = args.load_from
    cfg.finetune_from = args.finetune_from
    cfg.log_file = 'validate.txt' if args.validate else 'train.txt'


    # cfg.optimizer.betas = tuple(
    #   map(
    #       float,
    #       cfg.optimizer.betas.replace(' ','')[1:-2].split(',')))
    if cfg.isExists('optimizer.betas'):
        cfg.optimizer.betas = tuple(cfg.optimizer.betas)
    cfg.gpus = args.gpus
    return cfg


def parseArgs() -> argparse.Namespace:


    parser = argparse.ArgumentParser(description='Main Parse.')


    parser.add_argument(
        'config', type=str,
        help='Main config file in configs dir')
    parser.add_argument(
        '--work_dir', type=str, default='./logs/',
        help='The directory where save the preds result')
    parser.add_argument(
        '--gpus', nargs='+', type=int, default=[0],
        help="A list of available gpus like : [0 1]")
    parser.add_argument(
        '--validate', action='store_true',
        help='Evaluate the module')
    parser.add_argument(
        '--random_seed', type=int, default=0,
        help='Random seed for this program')
    parser.add_argument(
        '--load_from', default=None,
        help='the checkpoint file to resume from')
    parser.add_argument(
        '--finetune_from', default=None,
        help='whether to finetune from the checkpoint')
    parser.add_argument('--local_rank', type=int, default=0)


    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
