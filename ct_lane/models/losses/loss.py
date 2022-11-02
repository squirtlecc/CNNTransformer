from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.builder import LOSSES
import numpy as np
import cv2
from scipy.ndimage import convolve



class BootstrappingLoss(nn.Module):
    def __init__(self, alpha=0.5, thread=0.5):
        super(BootstrappingLoss, self).__init__()
        self._epsion = 1e-7
        self._alpha = alpha
        self.boostrap = self.getBoostrap(thread)
        self.ce = nn.BCELoss()
    
    def getBoostrap(self, thread):
        return lambda a,t,p: a*t + (1-a)*torch.where(p>thread, 0, 1).float()*t
        
    def forward(self, input, target):
        input_flat = input.reshape(-1)
        target_flat = target.reshape(-1)
        input_flat = torch.clamp(input_flat, self._epsion, 1.0-self._epsion)

        target_boot = self.boostrap(self._alpha, target_flat, input_flat)
        loss = self.ce(input_flat, target_boot)
        return loss


class FocalLoss(nn.Module):
    def __init__(self, gamma=4, alpha=None, normalized=False, reduction='mean', reduced_threshold=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.eps = 1e-10
        self.reduced_threshold = reduced_threshold
        self.normalized = normalized
        self.reduction = reduction
        """Compute binary focal loss between target and output logits.
        See :class:`~pytorch_toolbelt.losses.FocalLoss` for details.
        Args:
            output: Tensor of arbitrary shape (predictions of the model)
            target: Tensor of the same shape as input
            gamma: Focal loss power factor
            alpha: Weight factor to balance positive and negative samples. Alpha must be in [0...1] range,
                high values will give more weight to positive class.
            reduction (string, optional): Specifies the reduction to apply to the output:
                'none' | 'mean' | 'sum' | 'batchwise_mean'. 'none': no reduction will be applied,
                'mean': the sum of the output will be divided by the number of
                elements in the output, 'sum': the output will be summed. Note: :attr:`size_average`
                and :attr:`reduce` are in the process of being deprecated, and in the meantime,
                specifying either of those two args will override :attr:`reduction`.
                'batchwise_mean' computes mean loss per sample in batch. Default: 'mean'
            normalized (bool): Compute normalized focal loss (https://arxiv.org/pdf/1909.07829.pdf).
            reduced_threshold (float, optional): Compute reduced focal loss (https://arxiv.org/abs/1903.01347).
        References:
            https://github.com/open-mmlab/mmdetection/blob/master/mmdet/core/loss/losses.py
        """
    def forward(self, outputs, targets):

        outputs, targets = outputs.reshape(-1, 1), targets.reshape(-1, 1) # (N, 1)


        logpt = F.binary_cross_entropy(outputs, targets, reduction="none")
        # logpt = F.binary_cross_entropy_with_logits(output, target, reduction="none")
        pt = torch.exp(-logpt)

        # compute the loss
        if self.reduced_threshold is None:
            focal_term = (1.0 - pt).pow(self.gamma)
        else:
            focal_term = ((1.0 - pt) / self.reduced_threshold).pow(self.gamma)
            focal_term[pt < self.reduced_threshold] = 1

        loss = focal_term * logpt
        if self.alpha is not None:
            loss *= self.alpha * targets + (1 - self.alpha) * (1 - targets)

        if self.normalized:
            norm_factor = focal_term.sum().clamp_min(self.eps)
            loss /= norm_factor

        if self.reduction == "mean":
            loss = loss.mean()
        if self.reduction == "sum":
            loss = loss.sum()
        if self.reduction == "batchwise_mean":
            loss = loss.sum(0)

        return loss

# class FocalLoss(nn.Module):
#     '''nn.Module warpper for focal loss'''

#     def __init__(self, gamma=4):
#         super(FocalLoss, self).__init__()
#         self.neg_loss = self._neg_loss
#         self.eps = 1e-10
#         self.gamma = gamma

#     def _neg_loss(self, pred, gt, channel_weights=None):
#         ''' Modified focal loss. Exactly the same as CornerNet.
#         Runs faster and costs a little bit more memory
#         Arguments:
#         pred (batch x c x h x w)
#         gt_regr (batch x c x h x w)
#         '''

        # loss = 0
        # # one-hot 中等于1的为正样本
        # pos_inds = gt.eq(1).float()
        # # 小于1 eq(0) 为负样本
        # neg_inds = gt.lt(1).float()
        # # 负样本的权重
        # # 4 was gamma focusing parameter
        # # 调制系数(modulating factor)
        # # gamma = 0 就是传统CELoss
        # neg_weights = torch.pow(1 - gt, self.gamma)

        # pos_loss = torch.log(pred) * torch.pow(1 - pred, 2)
        # pos_loss *= pos_inds
        # neg_loss = torch.log(1 - pred) * torch.pow(pred, 2)
        # neg_loss *= neg_weights * neg_inds

        # num_pos = pos_inds.float().sum()
        # if channel_weights is None:
        #     pos_loss = pos_loss.sum()
        #     neg_loss = neg_loss.sum()
        # else:
        #     pos_loss_sum = 0
        #     neg_loss_sum = 0
        #     for i in range(len(channel_weights)):
        #         p = pos_loss[:, i, :, :].sum() * channel_weights[i]
        #         n = neg_loss[:, i, :, :].sum() * channel_weights[i]
        #         pos_loss_sum += p
        #         neg_loss_sum += n
        #     pos_loss = pos_loss_sum
        #     neg_loss = neg_loss_sum

        # # 'loss -' mean negtive
        # loss = loss - (pos_loss + neg_loss) / (num_pos + self.eps)
        # return loss

    # def forward(self, pred, target, weights_list=None):
    #     return self.neg_loss(pred, target, weights_list)


class RegL1KpLoss(nn.Module):
    def __init__(self):
        super(RegL1KpLoss, self).__init__()

    def forward(self, output, target, mask=None):
        if mask is None:
            mask = target
        
        loss = F.l1_loss(output * mask, target * mask, reduction='sum')
        mask = mask.bool().float()
        loss = loss / (mask.sum() + 1e-4)
        return loss


class IoULoss(nn.Module):
    def __init__(self, ignore_index=255):
        super(IoULoss, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, outputs, targets):
        # outputs = outputs.argmax(dim=1)
        # targets = targets.argmax(dim=1)
        mask = (targets != self.ignore_index).float()
        targets = targets.float()
        num = torch.sum(outputs*targets*mask)
        den = torch.sum(outputs*mask + targets*mask - outputs*targets*mask)
        return 1 - num / den

class AntiIoULoss(nn.Module):
    def __init__(self, ignore_index=255):
        super(AntiIoULoss, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, outputs, targets):
        classes = outputs.shape[1]

        targets = outputs.repeat(1,classes,1,1).detach().cpu() # repeat along channel[012]->[012012012]
        outputs = outputs.repeat_interleave(classes-1, dim=1) # repeat along channel[012]->[001122]

        ntargets = targets.numpy()
        delete_indexs = [i+i*classes for i in range(classes)]
        ntargets = np.delete(ntargets, delete_indexs, axis=1)
        targets = torch.from_numpy(ntargets).to(outputs.device)

        mask = (targets != self.ignore_index).float() # mask[012]->[120201]
        targets = targets.float()
        num = torch.sum(outputs*targets*mask)
        den = torch.sum(outputs*mask + targets*mask - outputs*targets*mask)
        return num / den

class GeneralizedSoftDiceLoss(nn.Module):
    def __init__(
            self, p=1, smooth=1,
            reduction='mean', weight=None, ignore_lb=255):
        super(GeneralizedSoftDiceLoss, self).__init__()
        self.p = p
        self.smooth = smooth
        self.reduction = reduction
        self.weight = None if weight is None else torch.tensor(weight)
        self.ignore_lb = ignore_lb

    def forward(self, probs, label):
        '''
        args: logits: tensor of shape (N, H, W)
        args: label: tensor of shape(N, H, W)
        '''
        # compute loss
        numer = torch.sum((probs*label), dim=(1, 2))
        denom = torch.sum(probs.pow(self.p)+label.pow(self.p), dim=(1, 2))
        if not self.weight is None:
            numer = numer * self.weight.view(1, -1)
            denom = denom * self.weight.view(1, -1)
        numer = torch.sum(numer, dim=0)
        denom = torch.sum(denom, dim=0)
        loss = 1 - (2*numer+self.smooth)/(denom+self.smooth)

        if self.reduction == 'mean':
            loss = loss.mean()
        return loss


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        # flatten label and prediction tensors
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)
        intersection = (inputs * targets).sum()
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        return 1 - dice


class FocalTverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalTverskyLoss, self).__init__()

    def forward(
            self, inputs, targets, smooth=1,
            alpha=.5, beta=.5, gamma=1.):
        
        # flatten label and prediction tensors
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)
        
        # True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
        
        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)
        FocalTversky = (1 - Tversky)**gamma
                       
        return FocalTversky


class HausdorffERLoss(nn.Module):
    """Binary Hausdorff loss based on morphological erosion"""

    def __init__(self, alpha=2.0, erosions=10, **kwargs):
        super(HausdorffERLoss, self).__init__()
        self.alpha = alpha
        self.erosions = erosions
        self.prepare_kernels()

    def prepare_kernels(self):
        cross = np.array([cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))])
        bound = np.array([[[0, 0, 0], [0, 1, 0], [0, 0, 0]]])

        self.kernel2D = cross * 0.2
        self.kernel3D = np.array([bound, cross, bound]) * (1 / 7)

    @torch.no_grad()
    def perform_erosion(
        self, pred: np.ndarray, target: np.ndarray
    ) -> np.ndarray:
        bound = (pred - target)**2

        if bound.ndim == 5:
            kernel = self.kernel3D
        elif bound.ndim == 4:
            kernel = self.kernel2D
        else:
            raise ValueError(f"Dimension {bound.ndim} is nor supported.")

        eroted = np.zeros_like(bound)
        erosions = []

        for batch in range(len(bound)):

            # debug
            erosions.append(np.copy(bound[batch][0]))

            for k in range(self.erosions):

                # compute convolution with kernel
                dilation = convolve(bound[batch], kernel, mode="constant", cval=0.0)

                # apply soft thresholding at 0.5 and normalize
                erosion = dilation - 0.5
                erosion[erosion < 0] = 0

                if erosion.ptp() != 0:
                    erosion = (erosion - erosion.min()) / erosion.ptp()

                # save erosion and add to loss
                bound[batch] = erosion
                eroted[batch] += erosion * (k + 1) ** self.alpha

        # image visualization in debug mode
        return eroted

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """
        Uses one binary channel: 1 - fg, 0 - bg
        pred: (b, 1, x, y, z) or (b, 1, x, y)
        target: (b, 1, x, y, z) or (b, 1, x, y)
        """
        assert pred.dim() == 4 or pred.dim() == 5, "Only 2D and 3D supported"
        assert (
            pred.dim() == target.dim()
        ), "Prediction and target need to be of same dimension"

        # pred = torch.sigmoid(pred)

        eroted = torch.from_numpy(
            self.perform_erosion(
                pred.cpu().detach().numpy(), target.cpu().detach().numpy())
        ).float()

        loss = eroted.mean()

        return loss


# testing DDice accuracy > CLLoss
# CE - Log(Dice)
class CEdDice(nn.Module):

    def __init__(self):
        super(CEdDice, self).__init__()
        self.ce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10.0]))

    def forward(self, input, target):
        finput = input.sigmoid().reshape(-1)
        # finput = input.reshape(-1)
        ftarget = target.reshape(-1)
        intersection = (finput * ftarget).sum()
        dice = (2.*intersection + 1.)/(finput.sum() + ftarget.sum() + 1.)
        return self.ce(input, target) - 0.05*torch.log(dice)


# CE + L1 loss 论文尝试
class CLLoss(nn.Module):

    def __init__(self):
        super(CLLoss, self).__init__()
        self.ce = nn.BCELoss(weight=torch.tensor([10.0]))
        self.l1 = nn.L1Loss()

    def forward(self, input, target):
        return self.ce(input.sigmoid(), target) + self.l1(input.sigmoid(), target)


# KL
class KLDLoss(nn.Module):
    def __init__(self):
        super(KLDLoss, self).__init__()
        self.kld = nn.KLDivLoss(reduction='mean')
        self.ce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10.0]))

    def forward(self, input, target):
        s_input = input.sigmoid().log()
        return self.kld(s_input, target) + self.ce(input, target)

class ExistLoss(nn.Module):
    def __init__(self, theta=1.0):
        super(ExistLoss, self).__init__()
        self.theta = theta
        self.ce = nn.BCEWithLogitsLoss()
    def forward(self, input, mask):

        gt_exist_lane = torch.zeros(input[0].shape[0], input[0].shape[1])
        for i in range(mask.shape[0]):
            index = torch.unique(mask[i]).tolist()
            gt_exist_lane[i, index] = 1
        gt_exist_lane = gt_exist_lane[:, 1:].to(input[0].device)
        return self.theta*self.ce(input[1], gt_exist_lane)

class NLLLoss(nn.Module):
    def __init__(self, ignore_index=-100, weight=None) -> None:
        super(NLLLoss, self).__init__()
        self.nll = nn.NLLLoss(ignore_index=ignore_index, weight=weight)
    def forward(self, input, target):
        return self.nll(torch.log(input), target.argmax(dim=1))

@LOSSES.registerModule
class VLaneLoss(nn.Module):
    def __init__(self, weights, cfg=None):
        super(VLaneLoss, self).__init__()
        self.cfg = cfg
        self.ce_loss = nn.BCEWithLogitsLoss()
        nl_weight = torch.ones(cfg.decoder.num_class)
        nl_weight[0] = 0.4

        pos_weight = torch.tensor([10.0])

        self.loss_func = nn.ModuleDict({
            'focal_loss': FocalLoss(),
            'l1_loss': RegL1KpLoss(),
            'ce_loss': nn.BCELoss(),
            'dice_loss': DiceLoss(),
            'ft_loss': FocalTverskyLoss(),
            'nll_loss': NLLLoss(ignore_index=255, weight=nl_weight),
            'antiiou_loss': AntiIoULoss(),
            'exist_loss': ExistLoss(),
        })
        self.weights: dict = {**weights}
        # init the weights here
        


    def forward(self, x, **kwargs):
        losses = {'total_loss': 0}
        # pred_mask = torch.clamp(x.sigmoid(), min=1e-4, max=1-1e-4)
        pred_mask = x[0] if isinstance(x, tuple) else x
        mask = kwargs['gt_mask'].long()


        gt = F.one_hot(mask, mask.max()+1).permute(0, 3, 1, 2)
        if gt.shape[1] < pred_mask.shape[1]:
            p2d = [0, 0, 0, 0, 0, pred_mask.shape[1] - gt.shape[1]]
            gt = F.pad(gt, p2d, 'constant', 0)

        gt_mask = gt.float()

        for func, weight in self.weights.items():
            if weight > 0:
                if func == 'exist_loss':
                    losses[func] = self.loss_func[func](x, mask)
                else:
                    losses[func] = self.loss_func[func](pred_mask, gt_mask)
                losses['total_loss'] += (weight * losses[func])

        return losses 
