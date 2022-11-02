import torch
import os

import torch.nn.functional

from .general_utils import getWorkDir, createDir


def saveModel(net, optim, cfg, epoch, is_best=False):
    logs_dir = getWorkDir(cfg)
    ckpts_dir = os.path.join(logs_dir, 'ckpts')
    if not os.path.exists(ckpts_dir):
        createDir(ckpts_dir)
    cfg.logs_dir = logs_dir

    # epoch = recorder.epoch
    ckpt_name = 'best' if is_best else epoch
    torch.save({
        'net': net.state_dict(),
        'optim': optim.state_dict(),
        # 'scheduler': scheduler.state_dict(),
        # 'recorder': recorder.state_dict(),
        # 'epoch': epoch
    }, os.path.join(ckpts_dir, f"{ckpt_name}.pth"))



def loadNetworkSpecified(net, model_dir, logger=None):
    pretrained_net = torch.load(model_dir)['net']
    net_state = net.state_dict()
    state = {}
    for k, v in pretrained_net.items():
        if k not in net_state.keys() or v.size() != net_state[k].size():
            if logger:
                logger.record('Skip weights: ' + k)
            continue
        state[k] = v
    net.load_state_dict(state, strict=False)


def loadNetwork(net, model_dir, finetune_from=None, device='cpu', logger=None):
    if finetune_from:
        if logger:
            logger.record('Finetune model from: ' + finetune_from)
        loadNetworkSpecified(net, finetune_from, logger)
        return
    # temp for load different device issues.
    pretrained_model = torch.load(model_dir, map_location=device)
    net.load_state_dict(pretrained_model['net'], strict=False)


def loadModel(net, cfg, logger=None):
    if not cfg.load_from and not cfg.finetune_from:
        return
    gpu = ','.join(str(gpu) for gpu in cfg.gpus)
    device = "cuda:{}".format(gpu) if cfg.gpus[0] > -1 else "cpu"
    loadNetwork(net, cfg.load_from, cfg.finetune_from, device, logger)


def f1(y_pred, y_true):
    epsilon = 1e-7
    tp = (y_true * y_pred).sum(dim=0).to(torch.float32)
    tn = ((1 - y_true) * (1 - y_pred)).sum(dim=0).to(torch.float32)
    fp = ((1 - y_true) * y_pred).sum(dim=0).to(torch.float32)
    fn = (y_true * (1 - y_pred)).sum(dim=0).to(torch.float32)

    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)

    f1 = 2 * (precision*recall) / (precision + recall + epsilon)
    f1 = f1.clamp(min=epsilon, max=1-epsilon)
    return 1 - f1.mean()

if __name__ == '__main__':
    print('__main__')