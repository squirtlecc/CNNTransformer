import torch

# from PIL import Image
import numpy as np

from scipy.interpolate import CubicSpline, interp1d, UnivariateSpline, InterpolatedUnivariateSpline

import cv2
import os
from tqdm import tqdm
import json

# import matplotlib.pyplot as plt
# import torch.nn.functional as F


class Seg2Lane(object):

    @staticmethod
    def coord_op_to_ip(x, y, sample_factor):
        # (160*scale, 88*scale) --> (160*scale, 88*scale+16=720) --> (1280, 720)
        if x is not None:
            x = x*sample_factor[0]
        if y is not None:
            y = y*sample_factor[1]
        return x, y


    @staticmethod
    def coord_ip_to_op(x, y, out_shape):
        # (1280, 720) --> (1280, 720-16=704) --> (1280/scale, 704/scale)
        if x is not None:
            x = int(x/out_shape[1])
        if y is not None:
            y = int((y-16)/out_shape[0])
        return x, y

    @staticmethod
    def get_lanes_tusimple(seg_out, out_shape=(720, 1280)):
        # fit cubic spline to each lane
        TUSIMPLE_SHAPE = [720, 1280]
        cut_height = TUSIMPLE_SHAPE[0] - out_shape[0]
        assert (cut_height <= 160), "Tusimple Datasets cut_height could't larger than 160"
        h_samples = list(range(160, 720, 10))
        cs = []
        lane_ids = np.unique(seg_out[seg_out > 0])
        for idx, t_id in enumerate(lane_ids):
            xs, ys = [], []
            for y_op in range(seg_out.shape[0]):
                x_op = np.where(seg_out[y_op, :] == t_id)[0]
                if x_op.size > 0:
                    x_op = np.mean(x_op)
                    xs.append(x_op / seg_out.shape[1] * out_shape[1])
                    ys.append(y_op / seg_out.shape[0] * out_shape[0])

            if len(xs) >= 10:
                cs.append(CubicSpline(ys, xs, extrapolate=False))
                # cs.append(interp1d(ys, xs))
            else:
                cs.append(None)

        # get x-coordinates from fitted spline
        lanes = []
        all_points = []
        for idx, t_id in enumerate(lane_ids):
            if cs[idx] is not None:
                x_out = cs[idx](np.array(h_samples)-cut_height)
                x_out[np.isnan(x_out)] = -2.0
                x_out = np.int32(x_out)
                lanes.append(x_out.tolist())
                points = np.c_[x_out, np.array(h_samples)].tolist()
                all_points.append(points)
            # else:
                # print("Lane too small, discarding...")
                # lanes.append([-2 for i in range(len(h_samples))])
        return all_points

    @staticmethod
    def getTusimpleLanes(preds, out_shape=None):
        lanes = []
        if out_shape is None:
            out_shape = (preds.shape[-2], preds.shape[-1])
        for i in range(preds.shape[0]):
            lane = Seg2Lane.get_lanes_tusimple(
                        preds[i], out_shape)
            lanes.append(lane)
        return lanes



#----------------

    @staticmethod
    def getTuSimpleFormatWithExists(probmaps, thresh=.5, resize_shape=(720, 1280), exists=None):
        lanes = []
        TUSIMPLE_HEIGHT = 720
        H_SAMPLE = list(range(160, 720, 10))

        probmaps = probmaps.cpu().numpy()
        probmaps = probmaps[1:, ...]
        cut_height = TUSIMPLE_HEIGHT - resize_shape[0]
        assert cut_height <= 240, "Cannot cut more than 240 pixels for culane."
        if exists is None:
            exists = [True for _ in probmaps]
        for probmap, exist in zip(probmaps, exists):
            if exist == 0:
                continue
            probmap = cv2.blur(probmap, (9, 9), borderType=cv2.BORDER_REPLICATE)

            coord =  []

            for y in H_SAMPLE:
                proj_y = round((y - cut_height) / resize_shape[0] * probmap.shape[0])
                line = probmap[proj_y]

                if np.max(line) < thresh:
                    coord.append([-2, y])
                    continue
                value = np.argmax(line)
                x = value / probmap.shape[1] * resize_shape[1]
                coord.append([x, y])

            if(sum(np.array(coord)[:,0] > 0) < 5):
                continue
            lanes.append(coord)

        return lanes



#----------------





    @staticmethod
    def get_lane(prob_map, y_px_gap, pts, thresh, k=0, resize_shape=None):
        """
        Arguments:
        ----------
        prob_map: prob map for single lane, np array size (h, w)
        resize_shape:  reshape size target, (H, W)

        Return:
        ----------
        coords: x coords bottom up every y_px_gap px, 0 for non-exist, in resized shape
        """
        if resize_shape is None:
            resize_shape = prob_map.shape
        H, W = resize_shape

        coords = np.zeros(pts)
        coords[:] = -2.0
        for i in range(pts):
            y = int((H - 10 - i * y_px_gap))
            if y < 0:
                break
            line = prob_map[y, :]
            find_pt = np.argmax(line)
            if line[find_pt] > thresh:
                coords[i] = int(find_pt)

            # find_pt = np.where(line > thresh)[0]
            # if len(find_pt) > 0:
            #     coords[i] = int(np.mean(find_pt))

        if (coords > 0).sum() < 4:
            coords = np.zeros(pts)
            coords[:] = -2.0

        coords = Seg2Lane.fix_gap(coords)
        return coords

    @staticmethod
    def is_short(lane) -> bool:
        start = [i for i, x in enumerate(lane) if x > 0]
        if not start:
            return True
        else:
            return False

    @staticmethod
    def fix_gap(coordinate):
        if any(x > 0 for x in coordinate):
            start = [i for i, x in enumerate(coordinate) if x > 0][0]
            end = [
                i for i,
                x in reversed(list(enumerate(coordinate))) if x > 0][0]
            lane = coordinate[start:end+1]
            if any(x < 0 for x in lane):
                gap_start = [i for i, x in enumerate(
                    lane[:-1]) if x > 0 and lane[i+1] < 0]
                gap_end = [
                    i+1 for i,
                    x in enumerate(lane[:-1]) if x < 0 and lane[i+1] > 0]
                gap_id = [i for i, x in enumerate(lane) if x < 0]
                if len(gap_start) == 0 or len(gap_end) == 0:
                    return coordinate
                for id in gap_id:
                    for i in range(len(gap_start)):
                        if i >= len(gap_end):
                            return coordinate
                        if id > gap_start[i] and id < gap_end[i]:
                            gap_width = float(gap_end[i] - gap_start[i])
                            lane[id] = int((id - gap_start[i]) / gap_width * lane[gap_end[i]] + (
                                gap_end[i] - id) / gap_width * lane[gap_start[i]])
                # if not all(x > 0 for x in lane):
                #     coordinate = CubicSpline(ys, xs, extrapolate=False)
                coordinate[start:end+1] = lane
            
        return coordinate

    @staticmethod
    def probmap2lane(
            seg_pred, resize_shape=None,
            smooth=True, y_px_gap=10, pts=56, thresh=0.5):
        """
        Arguments:
        ----------
        seg_pred:      np.array size (lane_num, h, w)
        resize_shape:  reshape size target, (H, W)
        smooth:      whether to smooth the probability or not
        y_px_gap:    y pixel gap for sampling
        pts:     how many points for one lane
        thresh:  probability threshold

        Return:
        ----------
        coordinates: [x, y] list of lanes, e.g.: [ [[9, 569], [50, 549]] ,[[630, 569], [647, 549]] ]
        """
        if resize_shape is None:
            resize_shape = seg_pred.shape[1:]  # seg_pred (5, h, w)
        max_lane, h, w = seg_pred.shape
        H, W = resize_shape
        # seg_pred = seg_pred.astype(np.uint8)
        coordinates = []

        for i in range(max_lane - 1):
            prob_map = seg_pred[i + 1]
            
            if resize_shape:
                prob_map = cv2.resize(
                    prob_map, (W, H), interpolation=cv2.INTER_LINEAR_EXACT)
            if smooth:
                prob_map = cv2.blur(
                    prob_map, (2, 2), borderType=cv2.BORDER_REFLECT)

            coords = Seg2Lane.get_lane(prob_map, y_px_gap, pts, thresh, k=i)

            if Seg2Lane.is_short(coords):
                continue

            # if i == 4 and (coords>0).sum()>0:
            #     import matplotlib.pyplot as plt
            #     print( (coords>0).sum())
            #     fig = plt.figure(figsize=(15 ,3))
            #     for j in range(seg_pred.shape[0]):
            #         plt.subplot(1, seg_pred.shape[0]+1, j+1)
            #         plt.imshow(seg_pred[j], cmap=plt.cm.hot, vmin=0, vmax=1)
            #     plt.colorbar()
            #     plt.show()
            # 
            # | y
            # |
            # v------>x x(width) cut height do not change
            # tusimple 数据逆序 是从160-710 的 不然计算会出错
            coords = [[coords[j], H - 10 - j * y_px_gap] if coords[j] > 0 else [-2, H - 10 - j * y_px_gap] for j in
                range(pts-1, -1, -1)]
            coordinates.append(coords)

        if len(coordinates) == 0:
            coords = np.zeros(pts)
            coordinates.append(
                [[coords[j], H - 10 - j * y_px_gap] if coords[j] > 0 else [-2, H - 10 - j * y_px_gap] for j in
                range(pts-1, -1, -1)])

        return coordinates
