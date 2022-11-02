import torch.nn as nn
from models.builder import NETS, buildBackbones, \
    buildNecks, buildCores, buildLoss, buildDecoders


@NETS.registerModule
class Detector(nn.Module):
    def __init__(self, cfg):
        super(Detector, self).__init__()
        self.cfg = cfg
        self.backbone = buildBackbones(cfg)
        self.neck = buildNecks(cfg)
        self.core = buildCores(cfg)
        self.loss = buildLoss(cfg)
        self.decoder = buildDecoders(cfg)

    def forward(self, batch):

        output = {}
        losses = {}

        feature = batch['img']
        if self.backbone:
            feature = self.backbone(feature)
        bf = feature

        if self.neck:
            feature = self.neck(feature)

        if self.core:
            feature = self.core(feature[2])


        if self.decoder:
            feature = self.decoder(feature, [bf[0], bf[1]])


        if self.training:
            losses = self.loss(feature, gt_mask=batch['gt_mask'])
            output.update(loss=losses)
        
        if 'exist_loss' in self.cfg.loss.weights.keys():
            output.update(feature=feature[0], exists=feature[1])
        else:
            output.update(feature=feature[0])

        return output
