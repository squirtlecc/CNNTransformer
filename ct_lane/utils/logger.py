import logging
import os
import cv2
import sys
import numpy as np
import torch
from .general_utils import getWorkDir, saveConfig, pngMaskJpg
from utils.general_utils import tensor2Imgs
from utils.tusimple_utils import Seg2Lane
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
from typing import Any
from collections import namedtuple
import tensorboard
from utils.summary import summary

from utils.calculate_fps import validateFPS
from torch.utils.tensorboard.writer import SummaryWriter

from threading import Lock
from fvcore.nn import FlopCountAnalysis, flop_count_table

class LoggerMeta(type):
    _instances = {}
    _lock: Lock = Lock()

    def __call__(cls, *args, **kwargs):
        """
        Possible changes to the value of the `__init__` argument do not affect
        the returned instance.
        """
        with cls._lock:
            if cls not in cls._instances:
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls] = instance
        return cls._instances[cls]


def initLogger(log_file=None, log_level=logging.INFO):
    stream_handle = logging.StreamHandler()
    handlers = list()
    handlers.append(stream_handle)

    if log_file is not None:
        file_handler = logging.FileHandler(log_file, 'w')
        handlers.append(file_handler)

    # formatter = logging.Formatter(
    #      "%(asctime)s / %(name)s [%(levelname)s] : %(message)s")
    formatter = logging.Formatter("[%(levelname)s] %(asctime)s : %(message)s", datefmt='%m-%d %H:%M')
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)

    logging.basicConfig(level=log_level, handlers=handlers)


class TensorBoardGraphs(torch.nn.Module):
    """
    Wrapper class for m odel with dict/list rvalues
    """
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_x) -> Any:
        """
        Wrap forward call.
        """
        data = self.model(input_x)

        if isinstance(data, dict):
            data = self.flat_dict(data)
            data_named_tuple = namedtuple(
                "ModelEndpoints", sorted(data.keys()))  # type: ignore
            data = data_named_tuple(**data)  # type: ignore
        elif isinstance(data, list):
            data = tuple(data)
        return data

    def flat_dict(self, ddict, dict_name=None):
        f_dict = {}
        for k in ddict.keys():
            fix_dict_name = ""
            if isinstance(ddict[k], dict):
                f_dict.update(self.flat_dict(ddict[k], k))
                continue
            if dict_name is not None:
                fix_dict_name = dict_name+"__"
            fix_dict_name = f"{fix_dict_name}{k}"
            f_dict.update({fix_dict_name: ddict[k]})
        return f_dict


class Logger(metaclass=LoggerMeta):
    def __init__(self, cfg):
        self.cfg = cfg
        
        self.logs_dir = getWorkDir(cfg)
        log_file = 'logs.txt'
        if cfg.haskey('log_file'):
            log_file = cfg.log_file
        self.logs_file = os.path.join(self.logs_dir, log_file)
        initLogger(self.logs_file)
        self.logging = logging.getLogger(__name__)
        self.logging.info('CONFIG : Saved Configs.')
        saveConfig(cfg)
        self.tensor_board_dir = os.path.join(self.logs_dir, 'tensorboard')
        self.tb = SummaryWriter(self.tensor_board_dir)
        self.epoch = 0
        self.step = 0
        self.loss = 0.
        self.batch_time = 0.
        self.data_time = 0.
        self.lr = 0.

    def record(self, message=None, log_level='info'):
        try_logging = getattr(self.logging, log_level)
        if message is None:
            try_logging(self)
        else:
            try_logging(message)

    def update(self, epoch=0):
        self.epoch = epoch + 1
        self.step += 1

    def logNet(self, net):
        # TODO: Bug cant save net to tensorboard
        # net_summary = net
        data_cfg = self.cfg
        if self.cfg.haskey('dataset'):
            data_cfg = self.cfg.dataset
        fake_data = {
            'img': torch.zeros(1, 3, data_cfg.img_height, data_cfg.img_width).cuda()*255.0,
            'gt_mask': torch.zeros(1, data_cfg.img_height, data_cfg.img_width).cuda()
        }
        net_summary = summary(net, fake_data)
        self.logging.info("\n"+net_summary)

        flops = FlopCountAnalysis(net, fake_data)
        # #self.logging.info("\n"+flop_count_table(flops))
        flop_count_table(flops)
        self.logging.info(f"All Models FLOPs: {flops.total()*1e-9}")
        # validateFPS(net, self, (self.cfg.img_height, self.cfg.img_width))

        # fake_data = {
        #     'img': torch.rand(8, 3, 320, 800).cuda()*255.0,
        #     'gt_mask': torch.rand(8, 320, 800).cuda()*3
        # }
        # self.tb.add_graph(TensorBoardGraphs(net), fake_data)

    def logValidate(self, acc):
        acc_result = acc

        self.tb.add_scalars('Validate/Accuracy', acc_result, self.epoch)

    def logNumber(self, name, number):
        self.tb.add_scalar(name, number, self.step)
    
    def logLearningRate(self, lr):
        self.lr = lr
        self.tb.add_scalar('LearningRate', lr, self.step)

    def logLoss(self, loss):
        self.loss += loss['total_loss'].item()
        save_loss = {}
        for k in loss.keys():
            scalar_loss = loss[k].item() if hasattr(loss[k], 'item') else loss[k]
            self.tb.add_scalar(f"Train/{k}", scalar_loss, self.step)
            save_loss.update({k: scalar_loss})
        self.tb.add_scalars('Train/Scalars', save_loss, self.step)

    def plotSomeImages(self, preds_mask, preds_lanes, data, thresh=0.5):
        self.logging.info("Saved Random Result Images to tensorBoard now.")
        # preds = output['feature'].argmax(dim=1)

        t = torch.arange(preds_mask.shape[1])
        t = t.reshape((1, preds_mask.shape[1], 1, 1))
        preds = preds_mask.cpu()*t
        preds = preds.sum(dim=1)
        oimg_w = self.cfg.dataset.original_img_width
        oimg_h = self.cfg.dataset.original_img_height
        img_norm = self.cfg.img_norm

        # data['img'] = tensor2Imgs(
        #     data['img'],
        #     mean=img_norm['mean'], std=img_norm['std'], to_rgb=False)
        plot_num = 5
        if preds.shape[0] < 5:
            plot_num = preds.shape[0]
        plt.figure(figsize=(15, 5))

        for i in range(plot_num):
            # plt.axis('off')
            plt.subplot(3, plot_num, i+1)
            plt.xticks([])  #去掉横坐标值
            plt.yticks([])  #去掉纵坐标值
            plt.title('pred')
            plt.imshow(preds[i].cpu().numpy())

            plt.subplot(3, plot_num, i+1+plot_num)
            plt.xticks([])  #去掉横坐标值
            plt.yticks([])  #去掉纵坐标值
            plt.title('gt')
            if 'gt_mask' in data.keys():
                plt.imshow(data['gt_mask'][i].cpu().numpy())

            plt.subplot(3, plot_num, i+1+plot_num*2)
            plt.title('img')
            plt.xticks([])  #去掉横坐标值
            plt.yticks([])  #去掉纵坐标值
            # 输入图片就转化过rgb 所以这里传false
            # img = tensor2Imgs(data['img'][i], to_rgb=False)
            raw_img = cv2.imread(data['img_metas'][i]['img_path'])
            raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
            real_img = cv2.resize(raw_img, (oimg_w, oimg_h))

            # for segmentation plot
            if isinstance(preds_lanes[i], str):
                # padding cut_height to match the original image
                seg = cv2.imread(preds_lanes[i])
                seg = cv2.cvtColor(seg, cv2.COLOR_BGR2GRAY)
                padding_preds = np.pad(seg,
                    ((0, 0), (0, self.cfg.cut_height)),
                    'constant', constant_values=0)
                mask_img = pngMaskJpg(real_img, padding_preds)
                plt.imshow(mask_img)
                continue
            plt.imshow(real_img)
            # plt pred point in real img
            # TODO: need fixed plot point( remove tusimpele utils )

            # pp 1
            # lane = Seg2Lane.get_lanes_tusimple(preds[i].cpu())
            # plt.title(f"lane_nums:{len(lane)}")
            # h_sample = list(range(160, 720, 10))
            # for j, l in enumerate(lane):
            #     for p in range(len(l)):
            #         if l[p] > 0:
            #             plt.plot(l[p], h_sample[p], 'o', color=f"C{j}")

            # pp 2
            # lane = Seg2Lane.probmap2lane(
            #     output['feature'][i].cpu().numpy()[:, :, :],
            #     resize_shape=(oimg_h, oimg_w), thresh=thresh)
            # plt.title(f"lane_nums:{len(lane)}")
            # plt.gca().invert_yaxis()

            for c, lane in enumerate(preds_lanes[i]):
                points = np.array([p for p in lane if p[0] > 0])
                points[:, 0] = np.clip(points[:, 0], 5, oimg_w-5)
                # lw lane width
                # ms marker scale
                plt.plot(
                    points[:, 0], points[:, 1],
                    marker='o', lw=0, ms=4, color=f"C{c}",
                    alpha=.4) if len(points) else None
            

            # 如果图画中有label需要打开legend
            # plt.legend(loc='best', prop = {'size':8})
        img_buffer = BytesIO()
        img_buffer.seek(0)
        plt.savefig(img_buffer, format='png')
        # plt.show() 后生成白板
        plt_img = np.array(Image.open(img_buffer))
        self.tb.add_image('Val_Images', plt_img, self.epoch, dataformats='HWC')
        plt.close()