from utils.register import Register, buildFromConfig

import torch.nn as nn
BACKBONES = Register('backbones')
NECKS = Register('necks')
CORES = Register('cores')
DECODERS = Register('decoders')
NETS = Register('nets')
LOSSES = Register('losses')



def build(cfgs, register, default_args=None):
    if isinstance(cfgs, list):
        modules = [
            buildFromConfig(cfg, register, default_args) for cfg in cfgs
        ]
        return nn.Sequential(*modules)
    else:

        return buildFromConfig(cfgs, register, default_args)


def buildBackbones(cfg, detail_args=None):
    return build(cfg.backbone, BACKBONES, default_args=detail_args)


def buildNecks(cfg, detail_args=None):
    return build(cfg.neck, NECKS, default_args=dict(cfg=cfg))


def buildCores(cfg, detail_args=None):
    return build(cfg.core, CORES, default_args=dict(cfg=cfg))


def buildNet(cfg):
    return build(cfg.net, NETS, default_args=dict(cfg=cfg))


def buildLoss(cfg):
    return build(cfg.loss, LOSSES, default_args=dict(cfg=cfg))


def buildDecoders(cfg, detail_args=None):
    return build(cfg.decoder, DECODERS, default_args=dict(cfg=detail_args))
