import os
import random
from tqdm import tqdm
from utils.general_utils import createDir
import cv2
import torch
import numpy as np
import utils.culane_metric as culane_metric
from utils.logger import Logger
from utils.culane_utils import getCULaneFormat

import utils.bdd100k_utils as smp

from .basic_dataset import BasicDataset
from .builder import DATASETS

LIST_FILE = {
    'train': 'list/train_gt.txt',
    'val': 'list/val_gt.txt',
    'test': 'list/test.txt',
} 

# not test

@DATASETS.registerModule
class BDD100K(BasicDataset):
    def __init__(self, data_root: str, mode: str, pipeline=None, dataset_cfg=None):
        super().__init__(data_root, mode, pipeline, dataset_cfg)
        # annotation files
        self.mode = mode

        self.anno_files = os.listdir(os.path.join(
            self.data_root, 'images', self.mode))
            
        self.data_infos = self._loadAnnotations()

        self.logger = Logger(self.cfg)

    def _loadAnnotations(self) -> list:
        # TODO: logger the data infos
        data_infos = []

        for anno_file in tqdm(self.anno_files, desc=f"Loading {self.mode}"):

            intact_img_path = os.path.join(
                self.data_root, 'images', self.mode, anno_file)
            intact_mask_path = intact_img_path.replace(
                '/images/', '/bdd_lane_gt/')
            intact_mask_path = intact_mask_path.replace('.jpg', '.png')
            data_infos.append({
                'img_name': anno_file,
                'img_path': intact_img_path,
                'mask_path': intact_mask_path,
            })

        if self.training:
            random.shuffle(data_infos)

        return data_infos

    def getFormatLanes(self, predictions, thresh=.5, resize_shape=(720, 1280), exists=list):
        batch_size = predictions.shape[0]
        batch_pred = []
        H, W = resize_shape
        resize_shape = (H-self.cfg.cut_height, W)
        output_basedir = os.path.join(self.cfg.work_dir, self.mode)
        createDir(output_basedir)
        for i in range(batch_size):
            pred = predictions[i].argmax(dim=0).cpu().detach().numpy()
            pred = cv2.resize(
                pred, (resize_shape[1], resize_shape[0]),
                interpolation=cv2.INTER_LINEAR_EXACT)
                # pred[thresh < thresh] = 0
            full_file_dir =  os.path.join(output_basedir, exists[i]['img_name'])
            cv2.imwrite(full_file_dir, pred)
            batch_pred.append(full_file_dir)
        return batch_pred

     # need fixed
    def evaluate(self, format_preds, output_basedir):
        # if self.mode == 'test':
        #     return 0
        output_basedir = os.path.join(output_basedir, self.mode)
        all_recall, all_f1, all_acc, all_iou = 0.0, 0.0, 0.0, 0.0
        for idx, pred in enumerate(tqdm(format_preds, desc='Evaluating')):
            label_seg = cv2.imread(self.data_infos[idx]['mask_path'], 0)
            # label_seg = np.expand_dims(label_seg, 0)
            pred = cv2.imread(pred, 0)
            pred = torch.from_numpy(pred).long().unsqueeze(0)
            label_seg = torch.from_numpy(label_seg).long().unsqueeze(0)
            label_seg[label_seg == 255] = 1

            tp_seg, fp_seg, fn_seg, tn_seg = smp.get_stats(pred, label_seg, mode='multiclass', threshold=None, num_classes=2)
            # f1 = smp.f1_score(tp_seg, fp_seg, fn_seg, tn_seg, reduction='none')
            accuracy = smp.balanced_accuracy(tp_seg, fp_seg, fn_seg, tn_seg, reduction="none")
            # recall = smp.recall(tp_seg, fp_seg, fn_seg, tn_seg, reduction="none")
            iou = smp.iou_score(tp_seg, fp_seg, fn_seg, tn_seg, reduction='none')
            # all_recall += recall[:,1]
            # all_f1 += f1[:,1]
            all_acc += accuracy[:,1]
            all_iou += iou[:,1]
        # acc = (all_acc / len(format_preds)).item()
        # f1 = (all_f1 / len(format_preds)).item()
        # recall = (all_recall / len(format_preds)).item()
        # iou = (all_iou / len(format_preds)).item()
        result = dict(
            acc = (all_acc / len(format_preds)).item(),
            # f1 = (all_f1 / len(format_preds)).item(),
            # recall = (all_recall / len(format_preds)).item(),
            iou = (all_iou / len(format_preds)).item()
        )

        print_result = f"{self.mode} {result}"
        # result = dict(acc=acc, recall=recall, f1=f1)
        self.logger.record(print_result)
        self.logger.logValidate(result)
        return result['acc']