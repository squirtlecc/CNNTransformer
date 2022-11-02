import random
import cv2
import torch
import math
import random

import tensorboard
from tqdm import tqdm
import numpy as np

from matplotlib import pyplot as plt
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F


from datasets.builder import buildDataloader
from utils.general_utils import getWorkDir, tensor2Imgs
from .builder import buildOptimizer, buildLogger, buildScheduler
from models.builder import buildNet
from utils.net_utils import saveModel, loadModel


class Mediator(object):
    def __init__(self, cfg):
        torch.manual_seed(cfg.random_seed)
        torch.cuda.manual_seed(cfg.random_seed)
        torch.cuda.manual_seed_all(cfg.random_seed)
        np.random.seed(cfg.random_seed)
        random.seed(cfg.random_seed)

        self.cfg = cfg
        self.logger = buildLogger(cfg)


        self.net = buildNet(cfg).cuda()
        self.logger.logNet(self.net)

        loadModel(self.net, cfg, self.logger)
        # if len(cfg.gpus) > -1:
        #     print(cfg.gpus)
        #     self.net = torch.nn.DataParallel(
        #         self.net, device_ids=cfg.gpus)
        #     torch.distributed.init_process_group(
        #         backend='nccl', init_method='env://',
        #         world_size=2, rank=2)
        #     self.net = DDP(
        #         self.net, device_ids=[0, 1])
        self.optimizer = buildOptimizer(cfg, self.net)
        


        self.val_loader = None

        self.metric = 0.

    def _2CUDA(self, batch) -> dict:
        for b in batch:
            if isinstance(batch[b], torch.Tensor):
                batch[b] = batch[b].cuda()
        return batch

    def trainEpoch(self, epoch: int, train_loader):
        self.net.train()
        logger_loss = 0.0
        loop = tqdm(train_loader, desc="Train-{}".format(epoch))
        # max_iter = len(train_loader)
        for i, data in enumerate(loop):


            data = self._2CUDA(data)

            output = self.net(data)


            self.optimizer.zero_grad()
            loss = output['loss']['total_loss']
            loss.backward()
            self.optimizer.step()

            self.scheduler.step()


            self.logger.update(epoch)
            self.logger.logLoss(output['loss'])
            self.logger.logLearningRate(lr=self.optimizer.param_groups[0]['lr'])
            tqdm_dict = {
                'loss': loss.item(),
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}",
            }
            loop.set_postfix(tqdm_dict)
            logger_loss = loss.item()
        self.logger.record(f"Epoch-{epoch}: lr:{self.optimizer.param_groups[0]['lr']:.2e} ,loss:{logger_loss:.4f}")
    def train(self):


        train_loader = buildDataloader(
            self.cfg.dataset.train, self.cfg.dataset, is_train=True)

        self.scheduler = buildScheduler(
            self.cfg, self.optimizer, len(train_loader))

        for epoch in range(self.cfg.epochs):
            self.trainEpoch(epoch, train_loader)
            self.saveCheckpoint(epoch)
            if not (epoch+1) % 1:
                self.validate(epoch)


    def validate(self, epoch=None):
        if not self.val_loader:
            self.val_loader = buildDataloader(
                self.cfg.dataset.val, self.cfg.dataset, is_train=False)
        self.net.eval()
        lane_thresh = 0.3
        preds = []
        img_metas = []
        r_save_val_serial = random.randint(0, len(self.val_loader)-1)
        r_save_data = dict(thresh=lane_thresh)


        exist_lane_t = 0
        exist_lane_all = 0

        loop = tqdm(self.val_loader, desc='Validate')
        for i, data in enumerate(loop):
            data = self._2CUDA(data)
            with torch.no_grad():
                output = self.net(data)


                preds_mask = output['feature']
                # test for existance of lane
                if 'exists' in output.keys():
                    exists = output['exists'] > 0.5
                    gt_exist_lane = torch.zeros(preds_mask.shape[0], preds_mask.shape[1])
                    for le in range(gt_exist_lane.shape[0]):
                        index = torch.unique(data['gt_mask'][le]).tolist()
                        gt_exist_lane[le, index] = 1
                    gt_exist_lane = gt_exist_lane[:, 1:]
                    exist_lane_all += gt_exist_lane.numel()
                    exist_lane_t +=  gt_exist_lane.numel() - (gt_exist_lane.int() ^ exists.cpu().int()).sum().item()
                else:
                    exists = None
                
                
                if self.cfg.dataset.val.type == 'BDD100K':
                    exists = data['img_metas']

                
                preds_lanes = self.val_loader.dataset.getFormatLanes(
                    preds_mask, thresh=lane_thresh, exists=exists)

                preds.extend(preds_lanes)
                img_metas.extend(data['img_metas'])

                if r_save_val_serial == i:
                    r_save_data.update(
                        preds_mask=preds_mask, preds_lanes=preds_lanes,
                        data=data)

        out = self.val_loader.dataset.evaluate(preds, getWorkDir(self.cfg))
        if out > self.metric and epoch is not None:
            self.metric = out
            self.saveCheckpoint(is_best=True)
        self.logger.record(f"Best models: {str(self.metric)}")

        if exist_lane_all != 0:
            self.logger.record(f"Lane exist acc: {exist_lane_t/exist_lane_all}") 


        if len(r_save_data.keys()) > 1 and epoch is not None:
            self.logger.plotSomeImages(**r_save_data)



    def saveCheckpoint(self, epoch=0, is_best=False):
        if (epoch % self.cfg.each_epoch_save == 0 and epoch != 0) or (epoch == self.cfg.epochs - 1):
            saveModel(self.net, self.optimizer, self.cfg, epoch, is_best)
        if is_best:
            saveModel(self.net, self.optimizer, self.cfg, -1, is_best)
