import os
import random
from tqdm import tqdm

import utils.culane_metric as culane_metric
from utils.logger import Logger
from utils.culane_utils import getCULaneFormat, getCULaneFormatWithExists

from .basic_dataset import BasicDataset
from .builder import DATASETS

LIST_FILE = {
    'train': 'list/train_gt.txt',
    'val': 'list/val_gt.txt',
    'test': 'list/test.txt',
} 

# not test

@DATASETS.registerModule
class CULane(BasicDataset):
    def __init__(self, data_root: str, mode: str, pipeline=None, dataset_cfg=None):
        super().__init__(data_root, mode, pipeline, dataset_cfg)
        # annotation files
        self.mode = mode
        self.anno_files = list()
        for data_list_file in dataset_cfg.data_list[mode]:
            all_list = list(open(os.path.join(data_root, data_list_file), 'r'))
            self.anno_files.extend(all_list)
            
        # list(open(
        #     os.path.join(data_root, dataset_cfg.data_list[mode][i]), 'r')
        #     for i in range(len(dataset_cfg.data_list[mode])))

        self.h_samples = list(range(230, 589, 20))
        self.data_infos = self._loadAnnotations()

        self.logger = Logger(self.cfg)

    def _loadAnnotations(self) -> list:
        # TODO: logger the data infos
        data_infos = []

        for anno_file in tqdm(self.anno_files, desc=f"Loading {self.mode}"):

            anno_file = anno_file.split()

            img_dir = anno_file[0]
            img_dir = img_dir[1 if img_dir[0] == '/' else 0::]
            intact_img_path = os.path.join(self.data_root, img_dir)
            if len(anno_file) > 1:
                mask_dir = anno_file[1]
                mask_dir = mask_dir[1 if mask_dir[0] == '/' else 0::]
                intact_mask_path = os.path.join(self.data_root, mask_dir)
            else:
                # TODO: need test this part if the mask is not exist in w16_test
                intact_mask_path = None
                intact_mask_path = os.path.join(self.data_root, 'laneseg_label_w16_test', img_dir)
                intact_mask_path = intact_mask_path.replace('.jpg', '.png')
            intact_lane_path = intact_img_path.replace('.jpg', '.lines.txt')
            with open(intact_lane_path, 'r') as lane_file:
                data = [list(
                    map(float, line.split())) for line in
                    lane_file.readlines()]

            lanes = [
                [(lane[i], lane[i + 1])
                    for i in range(0, len(lane), 2)
                    if lane[i] >= 0 and lane[i + 1] >= 0] for lane in data]
            # remove duplicated points
            lanes = [list(set(lane)) for lane in lanes]
            # remove lanes with less than 2 points
            lanes = [lane for lane in lanes if len(lane) > 3]
            # sort by y
            lanes = [sorted(lane, key=lambda x: x[1]) for lane in lanes]  

            data_infos.append({
                'img_name': img_dir,
                'img_path': intact_img_path,
                'mask_path': intact_mask_path,
                'lanes': lanes,
            })
            if intact_img_path == None:
                data_infos[-1].pop('mask_path')

        if self.training:
            random.shuffle(data_infos)

        return data_infos

    def getFormatLanes(self, predictions, thresh=.5, resize_shape=(590, 1640), exists=None):
        batch_size = predictions.shape[0]
        batch_lanes = []
        H, W = resize_shape
        resize_shape = (H-self.cfg.cut_height, W)
        for i in range(batch_size):

            lanes = getCULaneFormatWithExists(predictions[i], thresh, resize_shape, exists[i])

            batch_lanes.append(lanes)
        return batch_lanes

    # need fixed
    def evaluate(self, format_preds, output_basedir):
        # if self.mode == 'test':
        #     return 0
        output_basedir = os.path.join(output_basedir,  self.mode)
        for idx, pred in enumerate(tqdm(format_preds, desc='Evaluating')):
            output_filename = self.data_infos[idx]['img_name'][:-3] + 'lines.txt'
            output_filepath = os.path.join(output_basedir, output_filename)
            os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
            # output = self.get_prediction_string(pred)
            out = []
            # pred points flatten(CULane save format)
            for lane in pred:
                lane = [item for points in lane for item in points]
                lane_str = ' '.join(['{:}'.format(pts) for pts in lane])
                if lane_str != '':
                    out.append(lane_str)

            output = '\n'.join(out)
            
            with open(output_filepath, 'w') as out_file:
                out_file.write(output)

        data_list_file = self.cfg.data_list[self.mode][0]
        gt_list_path = os.path.join(self.data_root, data_list_file)
        result = culane_metric.eval_predictions(
            output_basedir, self.data_root, gt_list_path, official=True)

        print_result = result
        acc = result['Precision']
        f1 = result['F1']
        self.logger.record(print_result)
        list(map(result.pop, ['TP', 'FN', 'FP']))
        self.logger.logValidate(result)
        return f1