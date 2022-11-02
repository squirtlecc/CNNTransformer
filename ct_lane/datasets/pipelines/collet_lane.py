import cv2

import numpy as np

import matplotlib.pyplot as plt
from datasets.builder import PIPELINES

from utils.mask_creator import makeLLamasMask, makeTuSimpleMask, makeCULaneMask, makeCurveLanesMask


@PIPELINES.registerModule
class ColletLane(object):
    def __init__(
            self, keys, meta_keys, line_thickness=None, max_lane_num=None, cfg=None):
        self.keys = keys
        self.meta_keys = meta_keys
        self.line_width = line_thickness
        self.max_lane_num = max_lane_num

        self.cfg = cfg

    def _trimImg(self, img, cut_height=0):
        img = img[:, :, 0] if len(img.shape) > 2 else img
        img = img.squeeze()
        return img[cut_height:, :]

    # create a gt_mask from sample
    def target(self, sample) -> dict:
        # if line width exist in configs files
        # means u want make mask and use it.
        # if not, just used mask to gt_mask.
        if self.line_width is None:
            # dont need trim the mask.(trim img before the pipeline)
            sample['gt_mask'] = sample.pop('mask')
            # for bdd100k
            # if max(sample['gt_mask']) == 255:
            #     sample['gt_mask'] = np.sum(sample['gt_mask'], axis=0)
            #     sample['gt_mask'][sample['gt_mask'] > 0] = 1
            return sample

        data = sample.copy()
        # opencv issues , resize need change height and width
        # self.out_shape = (self.cfg.img_width, self.cfg.img_height)
        # # fixed the scale bug
        # self.scale = (float(self.cfg.img_height) / self.cfg.original_img_height,
        #     float(self.cfg.img_width) / self.cfg.original_img_width)

        self.orgin_shape = (self.cfg.original_img_height, self.cfg.original_img_width)
    
        if self.cfg.train.type == 'TuSimple':
            gt_lane = makeTuSimpleMask(
                data['lanes'], self.orgin_shape,
                max_lane=self.max_lane_num, line_width=self.line_width)
        elif self.cfg.train.type == 'LLAMAS':
            gt_lane = makeLLamasMask(
                data['label_path'], self.orgin_shape,
                max_lane=self.max_lane_num, thickness=self.line_width)
        elif self.cfg.train.type == 'CULane':
            gt_lane = makeCULaneMask(
                data['lanes'], self.orgin_shape,
                max_lane=self.max_lane_num, line_width=self.line_width)
        elif self.cfg.train.type == 'CurveLanes':
            img_shape = data['img'].shape
            gt_lane = makeCurveLanesMask(
                data['label_path'], img_shape=img_shape,
                out_shape=(720, 1280),
                max_lane=self.max_lane_num, line_width=self.line_width)
        else:
            assert False, "Can\'t make gt_mask for this type of dataset."
        gt_lane = self._trimImg(gt_lane, self.cfg.cut_height)

        data['gt_mask'] = gt_lane
        
        return data

    def __call__(self, data):
        results = {}
        metas = {}
        collet_data = self.target(data)

        for key in self.meta_keys:
            if key == 'img_shape':
                metas[key] = collet_data['img'].shape[:2]
            metas[key] = collet_data[key]
        results['img_metas'] = metas
        for key in self.keys:
            results[key] = collet_data[key]

        # if 'mask' in data.keys():
        #     results['gt_mask'] = data['mask']

        return results
