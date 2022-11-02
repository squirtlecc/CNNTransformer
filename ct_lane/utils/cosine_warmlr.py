import torch.optim as optim
import torch
import torch.nn as nn
import argparse
import math
from copy import copy
import matplotlib.pyplot as plt


class CosineAnnealingWarmbootingLR:
    # cawb learning rate scheduler: given the warm booting steps, calculate the learning rate automatically   

    def __init__(
            self, optimizer, epochs, each_epoch_len,
            eta_min=0.05,  steps=[], step_scale=0.8, lf=None, warmup_period=0, epoch_scale=1.0):
        self.warmup_iters = warmup_period * each_epoch_len if warmup_period < epochs else warmup_period
        self.optimizer = optimizer
        self.eta_min = eta_min
        self.iters = -1
        self.iters_batch = -1
        self.base_lr = [group['lr'] for group in optimizer.param_groups]
        self.step_scale = step_scale
        self.steps = self._setSteps(steps, epochs, each_epoch_len, warmup_period)

        self.gap = 0
        self.last_epoch = 0     
        self.lf = lf
        self.epoch_scale = epoch_scale
        
        # Initialize epochs and base learning rates
        for group in optimizer.param_groups:
            group.setdefault('initial_lr', group['lr'])

    def _setSteps(self, steps, epochs, each_epoch_len, warmup_period=0):
        if isinstance(steps, int):
            T_mul = steps
            steps = []
            warmup_epoch = int(warmup_period//each_epoch_len)+T_mul
            steps.append(warmup_epoch)
            
            while((steps[-1]*T_mul) < epochs):
                steps.append(steps[-1]*T_mul + T_mul)

        steps.sort()
        steps = [s*each_epoch_len for s in steps]
        total_iters = epochs * each_epoch_len
        steps = [warmup_period] + [i for i in steps if (i < total_iters and i > warmup_period)]   + [total_iters]    
        return steps
        
    def step(self, external_iter=None):
        self.iters += 1
        if external_iter is not None:
            self.iters = external_iter
        
        # cos warm boot policy
        iters = self.iters + self.last_epoch
        scale = 1.0
        for i in range(len(self.steps)-1):
            if (iters <= self.steps[i+1]):
                self.gap = self.steps[i+1] - self.steps[i]
                iters = iters - self.steps[i]

                if i != len(self.steps)-2:
                    self.gap += self.epoch_scale
                break
            scale *= self.step_scale
        
        if self.lf is None:
            for group, lr in zip(self.optimizer.param_groups, self.base_lr):
                group['lr'] = scale * lr * ((((1 + math.cos(iters * math.pi / self.gap)) / 2) ** 1.0) * (1.0 - self.eta_min) + self.eta_min)
        else:
            for group, lr in zip(self.optimizer.param_groups, self.base_lr):
                group['lr'] = scale * lr * self.lf(iters, self.gap)

        if self.iters < self.warmup_iters:
            # rate = self.iters_batch / self.warmup_iters
            rate = min(1.0, (self.iters+1) / self.warmup_iters)
            # rate = 1.0 - math.log((self.iters+1) / self.warmup_iters) / math.log(1/self.warmup_iters)
            for group, lr in zip(self.optimizer.param_groups, self.base_lr):
                group['lr'] = lr * rate
        return self.optimizer.param_groups[0]['lr']


if __name__ == "__main__":
    from tqdm import tqdm
    import numpy as np
    
    epoch = 200
    each_epoch_len = 3600
    optim = optim.SGD([{'params': [torch.randn(1, 1)]}], lr=0.2, momentum=0.9)
    y = []
    x = [-1]
    eta_min = 0
    lf = lambda iters, gap: eta_min + (1 - eta_min) * ((1 + math.cos(math.pi * iters / gap)) / 2)**2
    cawr = CosineAnnealingWarmbootingLR(
        optim, epoch, each_epoch_len,
        steps=3, step_scale=0.6, warmup_period=3, eta_min=0, lf=lf)

    for i in tqdm(range(epoch)):
        for j in range(each_epoch_len):
            optim.step()
            cawr.step()
            y.append(optim.param_groups[0]['lr'])
            x.append(x[-1]+1)
    
    del(x[0])
    t = np.array(y)
    print(f"min lr: {min(y)}")
    print(f"all < 1e-5: {np.sum(t<1e-4)}")
    plt.plot(x, y, marker='.', lw=0, ms=4, color=f"C{i}", alpha=.4)
    plt.savefig('/disk/vase/logs/cawr1.png')
