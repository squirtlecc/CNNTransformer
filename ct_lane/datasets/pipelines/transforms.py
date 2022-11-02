import random
import cv2
import numpy as np
import torch
import numbers
import collections
from PIL import Image
import copy
import matplotlib.pyplot as plt
from datasets.builder import PIPELINES
import albumentations as al


def to_tensor(data) -> torch.Tensor:
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.

    Args:
        data (torch.Tensor | numpy.ndarray | Sequence | int | float): Data to
            be converted.
    """

    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(f'type {type(data)} cannot be converted to tensor.')


@PIPELINES.registerModule
class ToTensor(object):
    """Convert some results to :obj:`torch.Tensor` by given keys.

    Args:
        keys (Sequence[str]): Keys that need to be converted to Tensor.
    """

    def __init__(self, keys=['img', 'mask'], cfg=None):
        self.keys = keys

    def __call__(self, sample):
        data = sample.copy()
        if len(sample['img'].shape) < 3:
            sample['img'] = np.expand_dims(sample['img'], -1)
        for key in self.keys:
            data[key] = to_tensor(sample[key])

        data['img'] = data['img'].permute(2, 0, 1)
        return data

    def __repr__(self):
        return self.__class__.__name__ + f'(keys={self.keys})'


@PIPELINES.registerModule
class Resize(object):
    def __init__(self, size, cfg=None):
        assert (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = tuple(size)

    def __call__(self, sample):

        for sample_key in sample.keys():
            if not isinstance(sample[sample_key], np.ndarray):
                continue
            sample[sample_key] = cv2.resize(
                sample[sample_key], self.size, interpolation=cv2.INTER_NEAREST_EXACT)

        return sample


@PIPELINES.registerModule
class RandomRotation(object):
    def __init__(
            self, degree=(-10, 10), keys=['img', 'mask'],
            interpolation=(cv2.INTER_LINEAR, cv2.INTER_NEAREST),
            padding=None, cfg=None):
        self.degree = degree
        self.interpolation = interpolation
        self.padding = padding
        self.keys = keys
        if self.padding is None:
            self.padding = [0, 0]

    def _rotate_img(self, img, map_matrix):
        h, w = img.shape[0:2]
        r_img = cv2.warpAffine(
            img, map_matrix, (w, h),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
            borderValue=self.padding)
        return r_img

    # def _rotate_mask(self, sample, map_matrix):
    #     if 'mask' not in sample:
    #         return
    #     h, w = sample['mask'].shape[0:2]
    #     sample['mask'] = cv2.warpAffine(
    #         sample['mask'], map_matrix, (w, h), flags=cv2.INTER_NEAREST,
    #         borderMode=cv2.BORDER_CONSTANT, borderValue=self.padding)

    def __call__(self, sample):
        v = random.random()
        if v < 0.5:
            degree = random.uniform(self.degree[0], self.degree[1])
            h, w = sample['img'].shape[0:2]
            center = (w / 2, h / 2)
            map_matrix = cv2.getRotationMatrix2D(center, degree, 1.0)
            for k in self.keys:
                sample[k] = self._rotate_img(sample[k], map_matrix)
        return sample


@PIPELINES.registerModule
class RandomBlur(object):
    def __init__(self, keys=['img'], prob=0.5, cfg=None):
        self.keys = keys
        self.prob = prob

    def __call__(self, sample):
        v = random.random()
        if v < self.prob:
            for k in self.keys:
                sample[k] = cv2.GaussianBlur(
                    sample[k], (5, 5), random.uniform(1e-6, 0.6))
        return sample


@PIPELINES.registerModule
class RandomHorizontalFlip(object):
    """Randomly horizontally flips the given numpy Image with a probability of 0.5
    """

    def __init__(self, prob=0.5, keys=['img', 'mask'], cfg=None):
        self.prob = prob
        self.keys = keys

    def __call__(self, sample):
        v = random.random()
        if v < self.prob:
            for k in self.keys:
                # copy() must need it
                sample[k] = np.fliplr(sample[k]).copy()
        return sample


@PIPELINES.registerModule
class AlTransforms(object):
    def __init__(self, transforms, keys=['img', 'mask'], cfg=None):
        self.keys = keys
        self.cfg = cfg
        self.additional_targets = self.setAddTarget(keys)
        self.univ_compose, self.replay_compose = self.getCompose(transforms)

    def getCompose(self, transforms):
        # univ_compose useless now
        # buz album can identify img and mask by addition_targets
        replay_trans = []
        univ_trans = []

        for t_name in transforms.keys():
            stand_transform = transforms[t_name].copy()
            need_replay = stand_transform.pop('replay', False)
            # if t_name == 'HorizontalFlip':
            #     print(need_replay)
            if 'transforms' in stand_transform.keys():
                stand_transform['transforms'] = [
                    getattr(al, trans_name)(
                        **stand_transform['transforms'][trans_name])
                    for trans_name in stand_transform['transforms'].keys()]
            params = stand_transform
            al_trans = getattr(al, t_name)(**params)
            if need_replay:
                replay_trans.append(al_trans)
            else:
                univ_trans.append(al_trans)
            
        if len(univ_trans) > 0:
            univ_compose = al.Compose(
                univ_trans, additional_targets=self.additional_targets)
        else:
            univ_compose = None
        if len(replay_trans) > 0:
            replay_compose = al.ReplayCompose(
                replay_trans, additional_targets=self.additional_targets)
        else:
            replay_compose = None
        return univ_compose, replay_compose

    def setAddTarget(self, keys):
        add_targets = dict()
        for key in keys:
            if 'img' in key or 'image' in key:
                add_targets.update({key: 'image'})
            if 'mask' in key or 'gt' in key:
                add_targets.update({key: 'mask'})
        return add_targets

    def synthAdata(self, sample):
        if 'image' in sample.keys():
            return sample
        sample_image_key = list(self.additional_targets.keys())[
            list(self.additional_targets.values()).index("image")]
        sample['image'] = sample.pop(sample_image_key)
        return sample

    def __call__(self, sample):
        # do not consider img_metas(not in additional_keys)
        # but albu need a core key:image for index
        # conversion a img type to al name{image}

        sample = self.synthAdata(sample)
        if self.univ_compose is not None:
            sample = self.univ_compose(**sample)
        sample_image_key = list(self.additional_targets.keys())[
                list(self.additional_targets.values()).index("image")]
        # re-conversion
        sample[sample_image_key] = sample.pop('image')

        if self.replay_compose is not None:
            self.replays = None
            for k in self.keys:
                data = self.replay_compose(image=sample[k])
                sample[k] = data['image']
                self.replays = data['replay']
                sample[k] = al.ReplayCompose.replay(
                    self.replays, image=sample[k])['image']

        return sample


@PIPELINES.registerModule
class Normalize(object):
    def __init__(self, img_norm, cfg=None):
        self.mean = np.array(img_norm['mean'], dtype=np.float32)
        self.std = np.array(img_norm['std'], dtype=np.float32)

    def __call__(self, sample):
        img = sample['img']
        if len(self.mean) == 1:
            img = img - np.array(self.mean)  # single channel image
            img = img / np.array(self.std)
        else:
            img = img - np.array(self.mean)[np.newaxis, np.newaxis, ...]
            img = img / np.array(self.std)[np.newaxis, np.newaxis, ...]
        sample['img'] = img

        return sample

    def imnormalize_(self, sample, mean, std, to_rgb=True):
        """Inplace normalize an image with mean and std.
        Args:
            img (ndarray): Image to be normalized.
            mean (ndarray): The mean to be used for normalize.
            std (ndarray): The std to be used for normalize.
            to_rgb (bool): Whether to convert to rgb.
        Returns:
            ndarray: The normalized image.
        """
        # cv2 inplace normalization does not accept uint8
        img = sample['img']
        assert img.dtype != np.uint8
        mean = np.float64(mean.reshape(1, -1))
        stdinv = 1 / np.float64(std.reshape(1, -1))
        if to_rgb:
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)  # inplace
        cv2.subtract(img, mean, img)  # inplace
        cv2.multiply(img, stdinv, img)  # inplace
        sample['img'] = img
        return sample


if __name__ == "__main__":
    print('00')