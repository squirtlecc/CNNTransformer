import os
import re
import cv2
import numpy as np
import yaml
import datetime
import time

from .config import Config


def createDir(path):
    if not os.path.isdir(path):
        # os.path.split('a/b/c') => ('a/b','c')
        createDir(os.path.split(path)[0])
    else:
        return
    if not os.path.isdir(path):
        os.mkdir(path)


def saveConfig(cfg, save_path=None):
    if save_path is None:
        save_path = os.path.join(getWorkDir(cfg), 'config.yml')
    with open(save_path, 'w') as dumpfile:
        if cfg.haskey('dataset'):
            dumpfile.write(yaml.dump(cfg.cfg_dict.to_dict()))
        else:
            print('config not complete. skip save it.')


def getWorkDir(cfg: Config) -> str:
    if cfg.haskey('logs_dir'):
        return cfg.logs_dir

    now = datetime.datetime.now().strftime('%m%d_%H%M%S')
    if cfg.haskey('load_from'):
        if cfg.load_from is not None:
            # just testing. Have any bugs just delete the if block
            # check is validate mode and real some load model
            search_path = re.findall(r'\d+_\d+', cfg.load_from, re.M | re.I)
            if len(search_path) > 0:
                now = search_path[-1]
        
    lr = cfg.optimizer.lr if cfg is not None else 0.
    bs = cfg.dataset.batch_size if cfg is not None else 0
    hyper_param_str = '_lr_%1.0e_b_%d' % (lr, bs)
    dataset_type = cfg.dataset.train.type if cfg is not None else 'Undifined'

    work_dir = os.path.join(
        cfg.dataset.work_dir, dataset_type, now + hyper_param_str)

    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
    cfg.dataset.work_dir = work_dir
    cfg.logs_dir = work_dir
    return work_dir


def pngMaskJpg(jpg, png, color="green"):
    color_list = {
        'blue': [255, 0, 0],
        'red': [0, 0, 255],
        'green': [0, 255, 0],
        'yellow': [44, 208, 245],
        'pink': [217, 100, 252],
    }

    empty_image = np.zeros_like(jpg)
    if isinstance(color, str):
        if color not in color_list.keys():
            color = 'green'
        color = color_list[color]
    empty_image[png == 1] = color

    jpg[png != 0] = empty_image[png != 0] * 0.5 + jpg[png != 0]*0.5

    return jpg



def tensor2Imgs(
        tensor, mean=(0, 0, 0), std=(1, 1, 1),
        is_batch=None, to_rgb=False) -> np.ndarray:
    """try convert tensor to numpy [b, h, w, 3]
        tensor shape:
        support color imgs or gray:
        [batch_size, 3, h, w] or [batch_size, h, w]
        or single imgs:
        [3, h, w] or [h, w]
        (your can specify whether it is batch or i just
        simple to choose one)
    """
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    imgs = []
    num_imgs = tensor.size(0)

    def _imgIsGray(img, is_batch=True) -> bool:
        if is_batch:
            is_gray = (len(img.shape) == 3)
        else:
            is_gray = (len(img.shape) == 2)
        return is_gray

    def _imdenormalize(img, mean, std, to_rgb=False):
        # https://github.com/open-mmlab/mmcv/blob/master/mmcv/image/photometric.py
        # assert img.dtype != np.uint8
        mean = mean.reshape(1, -1).astype(np.float64)
        std = std.reshape(1, -1).astype(np.float64)
        img = cv2.multiply(img, std)  # make a copy
        cv2.add(img, mean, img)  # inplace
        # img = np.multiply(img, std)
        # img = np.add(img, mean)
        if to_rgb:
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)  # inplace
        return img

    if is_batch is None:
        is_batch = False
        if len(tensor.shape) > 3:
            is_batch = True
        elif len(tensor.shape) == 3 and num_imgs != 3:
            # maybe tensor was gray imgs and batch is 3
            # i think its your issue, pls using is_batch=True
            is_batch = True

    # 是否灰度图 进行处理
    if not _imgIsGray(tensor, is_batch):
        tensor = tensor.transpose(-2, -1).transpose(-3, -1)
    else:
        if len(mean) != 1 or len(std) != 1:
            mean, std = np.array(0), np.array(1)

    tensor = tensor.cpu().numpy()

    # 是否为单图处理
    if not is_batch:
        img = tensor
        img = (_imdenormalize(
            img, mean, std, to_rgb=to_rgb))
        # if img full black means img value too small need * 255
        # if full white means img value too big
        img = (np.clip(img, a_min=0, a_max=255)).astype(np.uint8)
        return img

    for img_id in range(num_imgs):
        img = tensor[img_id, ...]
        img = (_imdenormalize(
            img, mean, std, to_rgb=to_rgb))
        img = (np.clip(img, a_min=0, a_max=255)).astype(np.uint8)
        imgs.append(np.ascontiguousarray(img))
    imgs = np.array(imgs)
    return imgs


class Timer:
    # example:
    # with TImer('takes {:.1f} seconds.'):
    #     time.sleep(1)
    def __init__(self, print_tmpl=None):
        self.print_tmpl = print_tmpl if print_tmpl else '{:.3f}'
        self.start_time = 0
        self._t_last = time.time()

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_value, exc_tb):
        print(self.print_tmpl.format(time.time() - self.start_time))

    def sinceLastCheck(self):
        during = time.time() - self._t_last
        self._t_last = time.time()
        return during


_g_timers = []


def checkTime(timer_id):
    # the timer_id was only key to find different timer
    if timer_id not in _g_timers:
        _g_timers[timer_id] = Timer()
        return 0
    else:
        return _g_timers[timer_id].sinceLastCheck()


if __name__ == "__main__":
    t = Timer()
    from config_wo_mmcv import Config
    cfg = Config.fromfile('configs/tusimple.yml')
    # saveConfig(cfg.text, './logs/config.yml')
    str_str = "/disk/vase/fw_ld/logs/ckpts/20211015_225517_lr_3e-04_b_8/99.pth"
    print(t)
