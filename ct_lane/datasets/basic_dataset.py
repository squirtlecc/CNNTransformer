import os
import cv2
from torch.utils.data import Dataset

from datasets.builder import DATASETS
from .pipelines.compose import ComposePipeline


@DATASETS.registerModule
class BasicDataset(Dataset):
    def __init__(self, data_root, mode, pipeline=None, dataset_cfg=None):
        self.cfg = dataset_cfg
        self.data_root = data_root
        self.training = 'train' in mode
        
        self.pipeline = ComposePipeline(pipeline, dataset_cfg)
        
        # data_infos must include img_path, or mask_path
        # if not mask path, need have mask_type,
        # the mask_type need create gt_mask in pipeline(collect_lane)
        self.data_infos = []

    def __len__(self):
        # self.infos 
        return len(self.data_infos)
    
    def __getitem__(self, index):

        data_info = self.data_infos[index]
        sample = data_info.copy()

        img = cv2.imread(data_info['img_path'])
        # if img is None:
        #     print(data_info['img_path'])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = _trimImg(img, self.cfg.cut_height, 'img')
        sample.update(img=img)

        # mask only training need, so if validate do not load it.
        # but some time validate need got origin img for display.
        # if self.training:
        if 'mask_path' in sample.keys():
            # same dataset haven't existed mask, need plot in pipleline(collect_lane)
            mask = cv2.imread(sample['mask_path'], cv2.IMREAD_UNCHANGED)
            mask[mask == 255] = 1 # ignore mask == 255
            mask = _trimImg(mask, self.cfg.cut_height, 'label')
            sample.update(mask=mask)
        
        sample = self.pipeline(sample)
        return sample

    # all lanes format must:batch_list[pred_lanes[lane[pt0[x, y], pt1[x, y+t],...]]]
    def getFormatLanes(self):
        pass

    def evaluate(self):
        pass

    def view(self):
        pass


def _trimImg(img, cut_height, img_type):
    if img_type == 'label':
        if len(img.shape) > 2:
            img = img[:, :, 0]
        img = img.squeeze()
        return img[cut_height:, :]

    return img[cut_height:, :, :]

