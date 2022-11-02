from typing import Optional
import cv2
import os
import numpy as np
import json
import random
from utils.tusimple_utils import Seg2Lane
from utils.mask_creator import makeTuSimpleMask
from utils.tusimple_metric import LaneEval
from .basic_dataset import BasicDataset
from .builder import DATASETS
from utils.logger import Logger
from tqdm import tqdm
# dataloader这一块负责简单的文件读取,
# 之后数据需要经过pipeline进一步处理
# 将图片变形后并分为img(输入到网络中的数据) 和 meta(真值)


@DATASETS.registerModule
class TuSimple(BasicDataset):

    def __init__(self, data_root: str, mode: str, pipeline=None, dataset_cfg=None):
        super().__init__(data_root, mode, pipeline, dataset_cfg)
        # annotation files
        self.mode = mode
        self.anno_files = dataset_cfg.data_list[mode]
        self.h_samples = list(range(160, 720, 10))
        self.data_infos = self._loadAnnotations()
        # have a couple issues if this class create looger fisrt
        # make sure you have cfg.dataset.logs_dir
        # if your found it bugs.
        self.logger = Logger(self.cfg)

    def _loadAnnotations(self) -> list:
        #TODO:logger
        data_infos = []
        max_lanes = 0

        for anno_file in tqdm(self.anno_files, desc='Loading Annotations'):
            intact_anno_file = os.path.join(self.data_root, anno_file)

            with open(intact_anno_file, 'r') as anno:
                annos = anno.readlines()
            for anno in annos:
                data = json.loads(anno)
                y_samples = data['h_samples']
                gt_lanes = data['lanes']
                mask_path = data['raw_file'].replace('clips', 'seg_label')[:-3] + 'png'
                lanes = []
                for gt_points in gt_lanes:
                    points = []
                    for (x, y) in zip(gt_points, y_samples):
                        if x > 0:
                            points.append((x, y))
                    
                    if len(points) > 0:
                        lanes.append(points)

                max_lanes = max(max_lanes, len(lanes))

                img_name = data['raw_file']
                img_path = os.path.join(self.data_root, img_name)
                mask_path = os.path.join(self.data_root, mask_path)
                data_infos.append({
                    'img_name': img_name,
                    'img_path': img_path,
                    'mask_path': mask_path,
                    'lanes': lanes,
                })

        if self.training:
            random.shuffle(data_infos)

        self.max_lanes = max_lanes
        return data_infos

    def _pred2Lanes(self, pred) -> list:
        lanes = np.array(pred).astype(int)
        lanes = lanes[:, :, 0]

        return lanes.tolist()

    def _pred2TusimpleFormat(self, index, pred, run_time) -> str:
        img_name = self.data_infos[index]['img_name']
        # lanes = self._pred2Lanes(pred)
        output = {
            'raw_file': img_name,
            'lanes': pred,
            'run_time': run_time
        }
        return json.dumps(output)

    def _savePredictions(self, predictions, file_name, run_times=None) -> None:
        if run_times is None:
            run_times = np.ones(len(predictions)) * 1.e-3

        each_pics_result = []
        for i, (pred, runtime) in enumerate(zip(predictions, run_times)):
            # tusimple format: only need x points
            if len(pred) > 0:
                pred = np.array(pred)[..., 0].tolist()
            lanes = self._pred2TusimpleFormat(i, pred, runtime)
            each_pics_result.append(lanes)
        with open(file_name, 'w') as output:
            output.write('\n'.join(each_pics_result))

    def evaluate(self, predictions, output_basedir, run_times=None) -> float:

        pred_file = os.path.join(output_basedir, 'tusimple_prediction.json')

        self._savePredictions(predictions, pred_file, run_times)
        all_acc = []
        result = {'Accuracy': 0.0,'FP': 0.0,'FN': 0.0,'F1': 0.0}
        for vt, val_data in enumerate(self.cfg.data_list[self.mode]):
            
            test_json_file = os.path.join(self.data_root, val_data)
            jresult, acc = LaneEval.bench_one_submit(pred_file, test_json_file)
            all_acc.append(acc)
            lresult = json.loads(s=jresult)
            for r in lresult:
                k, v = r['name'], r['value']
                result[k] = round((result[k]*vt + v*100)/(vt+1.0),3)
        self.logger.record(result)
        self.logger.logValidate(result)
        return result['F1']

    def evaludateTrainData(self, predictions, run_times=None) -> dict:
        if run_times is None:
            run_times = np.ones(len(predictions)) * 1.e-3
        # caculate the train data need skip image data enhance
        accuracy, fp, fn = 0, 0, 0
        for i, (pred, runtime) in enumerate(zip(predictions, run_times)):
            # lanes = json.loads(self._pred2TusimpleFormat(i, pred, runtime))
            lanes = self.data_infos[i]['lanes']
            gt_lane = []
            for l in lanes:
                i = 0
                new_lane = []
                l_h_start = l[0][1]
                l_h_end = l[-1][1]
                for h in self.h_samples:
                    if h < l_h_start:
                        new_lane.append(-2)
                        continue
                    if h > l_h_end:
                        new_lane.append(-2)
                        continue
                    new_lane.append(l[i][0])
                    i += 1
                gt_lane.append(new_lane)

            # gt_lanes = np.array(raw_lanes)[:, :, 0].tolist()
            a, p, n = LaneEval.bench(
                pred, gt_lane, self.h_samples, runtime)
            accuracy += a
            fp += p
            fn += n
        result = {
            'accuracy': accuracy / len(predictions),
            'fp': fp / len(predictions),
            'fn': fn / len(predictions)}
        return result

    def getFormatLanes(self, output, thresh=.5, resize_shape=(720, 1280), exists=None) -> list:
        batch_size = output.shape[0]
        lanes = []
        H, W = resize_shape
        resize_shape = (H-self.cfg.cut_height, W)
        exists = None
        if exists is None:
            exists = [[True for _ in range(output.shape[1]-1)] for _ in range(batch_size)]
        for i in range(batch_size):
            # another way to get tusimple line dot

            #---
            output[output < thresh] = 0
            if exists is None:
                exist = [True for _ in range(output.shape[1]-1)]
            else:
                exist = exists[i]
            for k, el in enumerate(exist, 1):
                if not el:
                    output[i, k, ...] = 0
                
            # there we got the line dot, but h_sample was (160-cut, 720-cut, 10)
            lane = Seg2Lane.get_lanes_tusimple(
                output[i].argmax(dim=0).cpu(), out_shape=resize_shape)
            #--- 
            
            # lane = Seg2Lane.getTuSimpleFormatWithExists(output[i], thresh, resize_shape, exists[i])
            
            lanes.append(lane)
        return lanes


