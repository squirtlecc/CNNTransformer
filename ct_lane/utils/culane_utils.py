import numpy as np
import torch
from scipy.interpolate import CubicSpline

import cv2
from scipy.interpolate import InterpolatedUnivariateSpline, UnivariateSpline
import numpy as np

CULANE_WIDTH = 1640
CULANE_HEIGHT = 590
H_SAMPLE = range(589, 230, -20)

def getCULaneFormat(prediction, resize_shape=(590, 1640), thresh=.5):
    # prediction was single batch
    # fit cubic spline to each lane
    cut_height = CULANE_HEIGHT - resize_shape[0]
    assert cut_height <= 240, "Cannot cut more than 240 pixels for culane."
    h_samples = range(590, 240, -10)
    # h_samples = range(590, 240, -10)
    cs = []
    all_points = []
    lanes = []
    pred_mask = None
    for lane_id, pred in enumerate(prediction):
        # pred was single channel of preds(it means single lane)
        # but lane_id = 0 was a mask of lane, so we need to ignore it
        if lane_id == 0: 
            pred_mask = 1 - pred
            pred_mask[pred_mask >= thresh] = 1
            pred_mask[pred_mask < thresh] = 0
            continue
        pred[pred_mask == 0] = 0
        # test

        pred[pred < thresh] = 0
        xs, ys = [], []
        for y_op, pred_y in enumerate(pred):
            x_op = torch.where(pred_y > 0)[0]
            if x_op.shape[0] > 0:
                x_op = torch.mean(x_op.float()).cpu().numpy()
                x_ip = x_op * (resize_shape[1]) / pred.shape[1]
                y_ip = y_op * (resize_shape[0]) / pred.shape[0]
                xs.append(x_ip)
                ys.append(y_ip)
        if len(xs) >= 10:
            cs.append(CubicSpline(ys, xs, extrapolate=False))
            all_points.append(list(zip(xs, ys)))
        else:
            cs.append(None)

    # get x-coordinates from fitted spline
    for idx in range(prediction.shape[0]-1):
        lane = []
        if cs[idx] is not None:
            # now the shape of img be cut, so h_samples is not correct
            # h_samples = range(590-cut_height, 240-cut_height, -10)
            # but we also need origin h_sample for validate, 
            # so only need add the cut_height
            # when we got _x.(because cut only used in H. x points is not changed)
            y_out = np.array(h_samples)
            # y_out = np.array(h_samples) - cut_height
            # lane.append([round(_x, 1), int(_y)])
            x_out = cs[idx](y_out-cut_height)
            for _x, _y in zip(x_out, y_out):
                if np.isnan(_x):
                    continue
                else:
                    lane.append([round(_x, 1), int(_y)])
            lanes.append(lane)

    return lanes
    # return lanes, all_points

def getCULaneFormatWithExists(probmaps, thresh=.5, resize_shape=(590, 1640), exists=None):
    lanes = []
    probmaps = probmaps.cpu().numpy()
    probmaps = probmaps[1:, ...]
    cut_height = CULANE_HEIGHT - resize_shape[0]
    assert cut_height <= 240, "Cannot cut more than 240 pixels for culane."
    if exists is None:
        exists = [True for _ in probmaps]
    for probmap, exist in zip(probmaps, exists):
        if exist == 0:
            continue
        probmap = cv2.blur(probmap, (9, 9), borderType=cv2.BORDER_REPLICATE)

        coord = []
        for y in H_SAMPLE:
            proj_y = round((y - cut_height) / resize_shape[0] * probmap.shape[0])
            line = probmap[proj_y]
            if np.max(line) < thresh:
                continue
            value = np.argmax(line)
            x = value / probmap.shape[1] * resize_shape[1]
            if x > 0:
                coord.append([x, y])
        if len(coord) < 5:
            continue

        coord = np.array(coord)
        # coord = np.flip(coord, axis=0)
        # coord[:, 0] /= CULANE_WIDTH
        # coord[:, 1] /= CULANE_HEIGHT
        # lanes.append(Lane(coord).to_array())
        lanes.append(coord)

    return lanes

def getCULaneFormat1(prediction, resize_shape=(590, 1640), thresh=.5):
    # prediction was single batch
    pred = prediction.argmax(dim=0).cpu().numpy()

    pred_ids = np.unique(pred[pred > 0])
    pred_out = np.zeros_like(pred)

    # sort lanes based on their size
    lane_num_pixels = [np.sum(pred == ids) for ids in pred_ids]
    ret_lane_ids = pred_ids[np.argsort(lane_num_pixels)[::-1]]
    # retain a maximum of 4 lanes
    if ret_lane_ids.size > 4:
        #  print("Detected more than 4 lanes")
        ret_lane_ids = ret_lane_ids[:4]

        # sort lanes based on their location
        lane_max_x = [
            np.median(np.nonzero(np.sum(
                pred == ids, axis=0))[0]) for ids in ret_lane_ids]
        ret_lane_ids = ret_lane_ids[np.argsort(lane_max_x)]

    # assign new IDs to lanes
    for i, r_id in enumerate(ret_lane_ids):
        pred_out[pred == r_id] = i+1

    cs = []
    lane_ids = np.unique(pred_out[pred_out > 0])
    for idx, t_id in enumerate(lane_ids):
        xs, ys = [], []
        for y_op in range(pred_out.shape[0]):
            x_op = np.where(pred_out[y_op, :] == t_id)[0]
            if x_op.size > 0:
                x_op = np.mean(x_op)
                x_ip = x_op*(resize_shape[1] / pred_out.shape[1])
                y_ip = y_op*(resize_shape[0] / pred_out.shape[0])
                np.clip(x_ip, 0, resize_shape[1])
                np.clip(y_ip, 0, resize_shape[0])
                xs.append(x_ip)
                ys.append(y_ip)
        if len(xs) >= 10:
            cs.append(CubicSpline(ys, xs, extrapolate=False))
        else:
            cs.append(None)
    # culane metric type
    lanes = []
    h_samples = range(590, 269, -10)
    for idx, t_id in enumerate(lane_ids):
        lane = []
        if cs[idx] is not None:
            y_out = np.array(h_samples)
            x_out = cs[idx](y_out)
            for _x, _y in zip(x_out, y_out):
                if np.isnan(_x):
                    continue
                else:
                    # lane += [round(_x, 2), int(_y)]
                    lane.append([round(_x, 2), int(_y)])
                
            lanes.append(lane)
    return lanes

class Lane:
    def __init__(self, points=None, invalid_value=-2., metadata=None):
        super(Lane, self).__init__()
        self.curr_iter = 0
        self.points = points
        self.invalid_value = invalid_value
        self.function = UnivariateSpline(points[:, 1], points[:, 0], k=min(3, len(points) - 1))
        self.min_y = points[:, 1].min() - 0.01
        self.max_y = points[:, 1].max() + 0.01

        self.sample_y = range(589, 230, -10)
        self.img_h = 590
        self.img_w = 1640

        self.metadata = metadata or {}

    def __repr__(self):
        return '[Lane]\n' + str(self.points) + '\n[/Lane]'

    def __call__(self, lane_ys):
        lane_xs = self.function(lane_ys)
        lane_xs[(lane_ys < self.min_y) | (lane_ys > self.max_y)] = self.invalid_value
        return lane_xs

    def to_array(self):
        # sample_y = cfg.sample_y
        # img_w, img_h = cfg.ori_img_w, cfg.ori_img_h
        sample_y = self.sample_y
        img_w, img_h = self.img_w, self.img_h
        ys = np.array(sample_y) / float(img_h)
        xs = self(ys)

        valid_mask = (xs >= 0) & (xs < 1)
        lane_xs = xs[valid_mask] * img_w
        lane_ys = ys[valid_mask] * img_h
        lane = np.concatenate((lane_xs.reshape(-1, 1), lane_ys.reshape(-1, 1)), axis=1)
        return lane

    def __iter__(self):
        return self

    def __next__(self):
        if self.curr_iter < len(self.points):
            self.curr_iter += 1
            return self.points[self.curr_iter - 1]
        self.curr_iter = 0
        raise StopIteration