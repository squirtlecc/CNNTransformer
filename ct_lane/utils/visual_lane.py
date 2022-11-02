from distutils import file_util
from lib2to3.pgen2.driver import Driver
import cv2
import os
import json
from tqdm import tqdm

# import utils.general_utils
# from mask_creator import makeCULaneMask

def saveLanes(img, lanes, show=False, save_file=None):
    for lane in lanes:
        for x, y in lane:
            if x <= 0 or y <= 0:
                continue
            x, y = int(x), int(y)
            cv2.circle(img, (x, y), 4, (255, 0, 0), 2)
    if show:
        cv2.imshow('view', img)
        cv2.waitKey(0)

    if save_file:
        utils.general_utils.createDir(os.path.dirname(save_file))
        cv2.imwrite(save_file, img)

def visualPreds(preds_file, save_file, data_root) -> None:
    _fps = 10
    _size = (720, 360)
    _m_color = [
                (255, 0, 0), (0, 255, 0), (0, 0, 255),
                (255, 255, 0), (0, 255, 255), (255, 0, 255),
                (250, 128, 114), (127, 255, 0), (0, 191, 255)]

    def _makeMask(label, mask, line_width=20):
        image = mask
        for index, line in enumerate(label):
            pt_end = 1
            pt_start = 0
            points = [(int(p[0]), int(p[1])) for p in line]
            sorted(points, key=lambda k: (k[1], k[0]))
            while pt_end < len(points):
                image = cv2.line(
                    mask, points[pt_start], points[pt_end],
                    thickness=line_width, color=_m_color[index], lineType=8)
                pt_start += 1
                pt_end += 1
        return image

    # videro_writer = cv2.VideoWriter(
    #     save_file, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), _fps, _size)


    videro_writer = cv2.VideoWriter(
        save_file, cv2.VideoWriter_fourcc('x264'), _fps, _size, True)

    with open(preds_file, 'r') as anno:
            annos = anno.readlines()
    loop = tqdm(annos)
    y_samples = list(range(160, 720, 10))
    for anno in loop:
        data = json.loads(anno)
        gt_lanes = data['lanes']
        img_path = data['raw_file']
        lanes = []
        for gt_points in gt_lanes:
            points = []
            for (x, y) in zip(gt_points, y_samples):
                if x > 0:
                    points.append((x, y))
            if len(points) > 0:
                lanes.append(points)

        img_path = os.path.join(data_root, img_path)
        mask = cv2.imread(img_path)
        pred = cv2.resize(_makeMask(lanes, mask), _size)
        videro_writer.write(pred)

def visualPredsCULane(root_path, pred_path, save_path) -> None:
    _fps = 10
    _size = (1640//2, 590//2)
    _m_color = [(13, 23, 227), (250, 128, 114),
                (127, 255, 0), (0, 191, 255)]

    def _makeMask(label, mask, line_width=20):
        image = mask
        for index, line in enumerate(label):
            pt_end = 1
            pt_start = 0
            points = [(int(p[0]), int(p[1])) for p in line]
            sorted(points, key=lambda k: (k[1], k[0]))
            while pt_end < len(points):
                image = cv2.line(
                    mask, points[pt_start], points[pt_end],
                    thickness=line_width, color=_m_color[index], lineType=8)
                pt_start += 1
                pt_end += 1
        return image
    ##
    file_list = os.path.join(root_path, 'list/test.txt')
    all_list = list(open(file_list, 'r'))
    print(len(all_list))
    driver_path_map = {}
    videro_writer = cv2.VideoWriter()
    for file in tqdm(all_list):
        file = file.split()[0][1:]
        dir_frame = file.split('/')[0]
        line_file = file.replace('.jpg', '.lines.txt')
        img_path = os.path.join(root_path, file)
        
        f = file[-9:].replace('.jpg', '.lines.txt')
        pred_file_path = os.path.join(pred_path, line_file)
        full_save_path = os.path.join(save_path, dir_frame) + '.avi'

        if dir_frame not in driver_path_map.keys():
            videro_writer.release()
            driver_path_map[dir_frame] = False
            videro_writer = cv2.VideoWriter(
                full_save_path, cv2.VideoWriter_fourcc(*'XVID'), _fps, _size, True)

        if driver_path_map[dir_frame]:
            videro_writer = cv2.VideoWriter(
                            full_save_path, cv2.VideoWriter_fourcc(*'XVID'), _fps, _size, True)
        
        with open(pred_file_path, 'r') as lane_file:
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
            
        
        mask = cv2.imread(img_path)
        cv2.putText(mask, file, (20, 20), cv2.FONT_HERSHEY_PLAIN, 2, (0,255,0), 1, cv2.LINE_AA)
        pred = cv2.resize(_makeMask(lanes, mask), _size)
        videro_writer.write(pred)
    videro_writer.release()

    
if __name__ == "__main__":
    root_path = '/home/yu/space/yuqiang/datasets/CULane/'
    # pred_path = '/disk1/vase/github/fw_ld/logs/CULane/0621_190506_lr_2e-02_b_8/test'
    pred_path = '/home/yu/space/yuqiang/github/fw_ld/logs/CULane/0824_122857_lr_3e-02_b_8/test'
    save_path = '/home/yu/space/yuqiang/github/'
    visualPredsCULane(root_path, pred_path, save_path)