import torch

import math
from utils.logger import Logger
from datasets.builder import buildDataloader
from utils.cosine_warmlr import CosineAnnealingWarmbootingLR
record_logger = None


def buildOptimizer(cfg, net):
    # params = []
    cfg_optimizer = cfg.optimizer.copy()

    cfg_type = cfg_optimizer.pop('type')
    assert cfg_type in dir(torch.optim)

    return getattr(torch.optim, cfg_type)(net.parameters(), **cfg_optimizer)


def buildScheduler(cfg, optimizer, lens=None):
    cfg_cp = cfg.scheduler.copy()
    cfg_type = cfg_cp.pop('type')
    train_loader_len = lens
    if lens is None:
        train_loader_len = len(buildDataloader(cfg.dataset.train, cfg.dataset, is_train=False))

    if cfg_type not in dir(torch.optim.lr_scheduler):
        if cfg_type == 'warmup':
            return buildWarmup(optimizer, train_loader_len*cfg.epochs,
                    batch_size=cfg.dataset.batch_size, **cfg_cp)
        elif cfg_type == 'CosineAnnealingWarmbootingLR':
            cfg_cp['epochs'] = cfg.epochs
            cfg_cp['each_epoch_len'] = train_loader_len
            return CosineAnnealingWarmbootingLR(optimizer, **cfg_cp)
        else:
            raise ValueError("{} is not defined.".format(cfg_type))

    _scheduler = getattr(torch.optim.lr_scheduler, cfg_type)
    # add it for stand scheduler (epoch to each step)
    cfg_cp.step_size *= train_loader_len
    return _scheduler(optimizer, **cfg_cp)


def buildWarmup(
        optimizer, total_step,
        warm_step=4200, batch_size=8, exp_gamma=10, func_type='exp'):
    # warmup_step
    warmup_step = warm_step // batch_size
    if func_type == 'linear':
        def lr_func(t): return t / warmup_step if t < warmup_step \
                else (1 - (t - warmup_step) / (total_step - warmup_step)) ** 0.9
        # l = lambda t: t / warmup_step if t < warmup_step \
        #         else (1 - (t - warmup_step) / (total_step - warmup_step)) ** 0.9
    elif func_type == 'cos':
        lr_func = lambda x: ((1 + math.cos(x * math.pi / total_step)) / 2) * (1 - 0.2) + 0.2
    elif func_type == 'exp':
        # exponential learning rate decay
        lr_func = lambda t: t / warmup_step if t < warmup_step \
            else (math.exp(exp_gamma*(warmup_step - t)/(total_step)))
    else:
        raise ValueError("{} is not defined.".format(func_type))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_func)

def buildLogger(cfg):
    return Logger(cfg)
