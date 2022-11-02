import cv2
import numpy as np
import json
from scipy.interpolate import CubicSpline
# from .spline_creator import get_horizontal_values_for_four_lanes
from utils.llamas_utils import get_horizontal_values_for_four_lanes

def coord_op_to_ip(x, y, scale):
    # (160*scale, 88*scale) --> (160*scale, 88*scale+13) --> (1280, 717) --> (1276, 717)
    if x is not None:
        x = scale*x
        x = int(x*1276./1280.)
    if y is not None:
        y = int(scale*y+13)
    return x, y

def coord_ip_to_op(x, y, scale):
    # (1276, 717) --> (1280, 717) --> (1280, 717-13=704) --> (1280/scale, 704/scale)
    if x is not None:
        x = x*1280./1276.
        x = int(x/scale)
    if y is not None:
        y = int((y-13)/scale)
    return x, y

def match_multi_class(pred):
    pred_ids = np.unique(pred[pred > 0]) # find unique pred ids
    pred_out = np.zeros_like(pred) # initialize output array

    # return input array if no lane points in prediction/target
    if pred_ids.size == 0:
        return pred

    # sort lanes based on their size
    lane_num_pixels = [np.sum(pred == ids) for ids in pred_ids]
    ret_lane_ids = pred_ids[np.argsort(lane_num_pixels)[::-1]]
    # retain a maximum of 4 lanes
    if ret_lane_ids.size > 4:
        print("Detected more than 4 lanes")
        ret_lane_ids = ret_lane_ids[:4]
    elif ret_lane_ids.size < 4:
        print("Detected fewer than 4 lanes")

    # sort lanes based on their location
    lane_max_x = [np.median(np.nonzero(np.sum(pred == ids, axis=0))[0]) for ids in ret_lane_ids]
    ret_lane_ids = ret_lane_ids[np.argsort(lane_max_x)]

    # assign new IDs to lanes
    for i, r_id in enumerate(ret_lane_ids):
        pred_out[pred == r_id] = i+1

    return pred_out

def get_lanes_culane(seg_out, samp_factor):
    # fit cubic spline to each lane
    h_samples = range(300, 717, 1)
    cs = []
    lane_ids = np.unique(seg_out[seg_out > 0])
    for idx, t_id in enumerate(lane_ids):
        xs, ys = [], []
        for y_op in range(seg_out.shape[0]):
            x_op = np.where(seg_out[y_op, :] == t_id)[0]
            if x_op.size > 0:
                x_op = np.mean(x_op)
                x_ip, y_ip = coord_op_to_ip(x_op, y_op, samp_factor)
                xs.append(x_ip)
                ys.append(y_ip)
        if len(xs) >= 10:
            cs.append(CubicSpline(ys, xs, extrapolate=False))
        else:
            cs.append(None)
    # get x-coordinates from fitted spline
    lanes = []
    for idx, t_id in enumerate(lane_ids):
        lane = []
        if cs[idx] is not None:
            y_out = np.array(h_samples)
            x_out = cs[idx](y_out)
            for _x, _y in zip(x_out, y_out):
                if np.isnan(_x):
                    continue
                else:
                    lane += [_x, _y]
            lanes.append(lane)

    return lanes

# need to be updated
def get_llamas(seg_out, samp_factor):
    # fit cubic spline to each lane
    h_samples = range(300, 717, 1)
    cs = []
    lane_ids = np.unique(seg_out[seg_out > 0])
    for idx, t_id in enumerate(lane_ids):
        xs, ys = [], []
        for y_op in range(seg_out.shape[0]):
            x_op = np.where(seg_out[y_op, :] == t_id)[0]
            if x_op.size > 0:
                x_op = np.mean(x_op)
                x_ip, y_ip = coord_op_to_ip(x_op, y_op, samp_factor)
                xs.append(x_ip)
                ys.append(y_ip)
        if len(xs) >= 10:
            cs.append(CubicSpline(ys, xs, extrapolate=False))
        else:
            cs.append(None)
    # get x-coordinates from fitted spline
    lanes_str = ['l1', 'l0', 'r0', 'r1']
    lanes = dict()
    for idx, t_id in enumerate(lane_ids):
        lane = []
        if cs[idx] is not None:
            y_out = np.array(h_samples)
            x_out = cs[idx](y_out)
            for _x, _y in zip(x_out, y_out):
                if np.isnan(_x):
                    lane += [-1]
                else:
                    lane += [_x]
            lanes[lanes_str[idx]] = lane
    return lanes


def makeLLamasMask(
        json_path: str, ori_shape=(717, 1276),
        thickness=16, max_lane=5) -> np.ndarray:
    """ Creates pixel-level label of markings color coded by lane association
    Only for the for closest lane dividers, i.e. l1, l0, r0, r1
    Parameters
    ----------
    json_path: str
               path to label file
    Returns
    -------
    numpy.array
        pixel level segmentation with interpolated lanes (717, 1276)
    Notes
    -----
    Only draws 4 classes, can easily be extended for to a given number of lanes
    """
    # initialize output array
    seg = np.zeros(ori_shape, dtype=np.uint8)
    # get projected lane centers
    lanes = get_horizontal_values_for_four_lanes(json_path)

    for r in range(ori_shape[0]-1):
        for i, lane in enumerate(lanes):
            if max_lane < i:
                break
            if lane[r] >= 0 and lane[r+1] >= 0:
                # similar to CULane, draw lines with 16 pixel width
                seg = cv2.line(
                    seg, (round(lane[r]), r), (round(lane[r+1]), r+1),
                    i+1, thickness=thickness)
    return seg


def makeCULaneMask1(
        label, img_shape=(590, 1640),
        out_shape=None, max_lane=4, line_width=15):

    seg_mask = np.zeros(img_shape, dtype=np.uint8)
    for i, coords in enumerate(label):
        if len(coords) < 4:
            continue
        for j in range(len(coords)-1):
            pt_start = (int(coords[j][0]), int(coords[j][1]))
            pt_end = (int(coords[j+1][0]), int(coords[j+1][1]))
            seg_mask = cv2.line(
                seg_mask, pt_start, pt_end, i+1, line_width)
    
    if out_shape is not None:
        seg_mask = cv2.resize(
            seg_mask, out_shape, interpolation=cv2.INTER_NEAREST_EXACT)

    return seg_mask




def makeCULaneMask(
        label, img_shape=(590, 1640),
        out_shape=None, max_lane=4, line_width=15):
    # ---------- clean and sort lanes -------------
    lanes = []
    _lanes = []
    slope = []
    # identify 1st, 2nd, 3rd, 4th lane through slope
    # if max_lane is odd then left max = i, right max = i+1
    for line in label:
        points = line
        if len(points) > 1:
            _lanes.append(points)
            slope.append(np.arctan2(
                points[-1][1]-points[0][1], points[0][0]-points[-1][0]
                ) / np.pi * 180)

    _lanes = [_lanes[i] for i in np.argsort(slope)]
    slope = [slope[i] for i in np.argsort(slope)]

    lane_idx = [-1 for _ in range(max_lane)]

    for index in range(len(slope)):
        if slope[index] <= 90:
            for ld in range(index+1):
                if (max_lane//2)-1-ld < 0:
                    break
                lane_idx[(max_lane//2)-1-ld] = index-ld
        else:
            for ld in range(len(slope)-index):
                if (max_lane//2)+ld >= len(lane_idx):
                    break
                lane_idx[(max_lane//2)+ld] = index+ld
            break

    for idx in lane_idx:
        lanes.append([] if idx < 0 else _lanes[idx])

    seg_mask = np.zeros(img_shape, dtype=np.uint8)

    for i, coords in enumerate(lanes):
        if len(coords) < 4:
            continue
        for j in range(len(coords)-1):
            pt_start = (int(coords[j][0]), int(coords[j][1]))
            pt_end = (int(coords[j+1][0]), int(coords[j+1][1]))
            seg_mask = cv2.line(
                seg_mask, pt_start, pt_end, i+1, line_width)

    if out_shape is not None:
        seg_mask = cv2.resize(
            seg_mask, out_shape, interpolation=cv2.INTER_NEAREST_EXACT)

    return seg_mask


def makeTuSimpleMask(
        label, img_shape=(720, 1280),
        out_shape=None, max_lane=5, line_width=15):
    # ---------- clean and sort lanes -------------
    lanes = []
    _lanes = []
    slope = []
    # identify 1st, 2nd, 3rd, 4th lane through slope
    # if max_lane is odd then left max = i, right max = i+1
    for line in label:
        points = [(p[0], p[1]) for p in line]
        if len(points) > 1:
            _lanes.append(points)
            slope.append(np.arctan2(
                points[-1][1]-points[0][1], points[0][0]-points[-1][0]
                ) / np.pi * 180)

    _lanes = [_lanes[i] for i in np.argsort(slope)]
    slope = [slope[i] for i in np.argsort(slope)]

    lane_idx = [-1 for _ in range(max_lane)]

    for index in range(len(slope)):
        if slope[index] <= 90:
            for ld in range(index+1):
                if (max_lane//2)-1-ld < 0:
                    break
                lane_idx[(max_lane//2)-1-ld] = index-ld
        else:
            for ld in range(len(slope)-index):
                if (max_lane//2)+ld >= len(lane_idx):
                    break
                lane_idx[(max_lane//2)+ld] = index+ld
            break

    for idx in lane_idx:
        lanes.append([] if idx < 0 else _lanes[idx])

    seg_mask = np.zeros(img_shape, dtype=np.uint8)

    for i, coords in enumerate(lanes):
        if len(coords) < 4:
            continue
        for j in range(len(coords)-1):
            seg_mask = cv2.line(
                seg_mask, coords[j], coords[j+1], i+1, line_width)

    if out_shape is not None:
        seg_mask = cv2.resize(
            seg_mask, out_shape, interpolation=cv2.INTER_NEAREST_EXACT)

    return seg_mask


def makeCurveLanesMask(
        label, img_shape=(720, 1280),
        out_shape=None, max_lane=9, line_width=5):
    label_info = json.load(open(label))
    lanes_num = 0
    out_shape = img_shape if out_shape is None else out_shape

    all_points = []
    for index, line in enumerate(label_info['Lines']):
        lanes_num += 1
        # print(line)
        points_x = []
        points_y = []
        # get points
        scale = [out_shape[1] / (img_shape[1]+0.0),
                 out_shape[0] / (img_shape[0]+0.0)]
        for point in line:
            points_x.append(int(
                float(point['x']) * scale[0]))
            points_y.append(int(
                float(point['y']) * scale[1]))
        
        points = list(zip(points_x, points_y))
        # sort along y
        sorted(points, key=lambda k: (k[1], k[0]))
        all_points.append(points)
    # ---------- clean and sort lanes -------------
    lanes = []
    _lanes = []
    slope = []
    # identify 1st, 2nd, 3rd, 4th lane through slope
    # if max_lane is odd then left max = i, right max = i+1
    for line in all_points:
        points = line
        if len(points) > 1:
            _lanes.append(points)
            slope.append(abs(np.arctan2(
                points[-1][1]-points[0][1], points[0][0]-points[-1][0]
                ) / np.pi * 180))
    # print(slope)
    _lanes = [_lanes[i] for i in np.argsort(slope)]
    slope = [slope[i] for i in np.argsort(slope)]
    lane_idx = [-1 for _ in range(max_lane)]

    for index in range(len(slope)):
        if slope[index] <= 90:
            for ld in range(index+1):
                if (max_lane//2)-1-ld < 0:
                    break
                lane_idx[(max_lane//2)-1-ld] = index-ld
        else:
            for ld in range(len(slope)-index):
                if (max_lane//2)+ld >= len(lane_idx):
                    break
                lane_idx[(max_lane//2)+ld] = index+ld
            break

    for idx in lane_idx:
        lanes.append([] if idx < 0 else _lanes[idx])

    seg_mask = np.zeros(out_shape, dtype=np.uint8)

    for i, coords in enumerate(lanes):
        for j in range(len(coords)-1):
            pt_start = (int(coords[j][0]), int(coords[j][1]))
            pt_end = (int(coords[j+1][0]), int(coords[j+1][1]))
            seg_mask = cv2.line(
                seg_mask, pt_start, pt_end, i+1, line_width+int(line_width*scale[0]))

    return seg_mask

if __name__ == "__main__":
    import os
    # data_dir = "/disk/lane-datasets/LLAMAS"
    # lp = "labels/valid/images-2014-12-22-12-35-10_mapping_280S_ramps/1419280602_0873282000.json"
    # lp_dir = os.path.join(data_dir, lp)
    # seg_out = makeLLamasMask(lp_dir)
    data_dir = '/disk/lane-datasets/Curvelanes/'
    lp = 'valid/labels/c105ddad0167f20c619121e28a2c573c.lines.json'
    lp = os.path.join(data_dir, lp)
    img_shape = cv2.imread(lp.replace('.lines.json', '.jpg').replace('labels', 'images')).shape[:2]
    seg_out = makeCurveLanesMask(lp, img_shape=img_shape, out_shape=(720, 1280), line_width=5)
    print(seg_out.shape)
    # print(seg_out.shape)
    print(np.unique(seg_out, return_counts=True))