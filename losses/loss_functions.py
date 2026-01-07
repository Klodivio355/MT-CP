# This code is referenced from 
# https://github.com/facebookresearch/astmt/
# 
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# 
# License: Attribution-NonCommercial 4.0 International

import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.nn.modules.module import Module 
import numpy as np 

class FocalLoss(Module):
    def __init__(self):
        super(FocalLoss, self).__init__()
        ##self.ignore_index = ignore_index
        self.softmax = nn.LogSoftmax(dim=1)
        self.gamma=2
        self.weight=None
        self.alpha=0.25
        self.ignore_index = 255
        self.weight = torch.tensor([self.alpha, 1-self.alpha], device='cuda')
        self.nllLoss = nn.NLLLoss(weight=self.weight)

    def forward(self, input_tensor, target_tensor):
        log_prob = F.log_softmax(input_tensor, dim=-1)
        prob = torch.exp(log_prob)
        return F.nll_loss(
            ((1 - prob.squeeze()) ** self.gamma) * log_prob, 
            target_tensor.squeeze(), 
            weight=self.weight,
            reduction = "none"
        )

class SoftMaxwithLoss(Module):
    """
    This function returns cross entropy loss for semantic segmentation
    """

    def __init__(self):
        super(SoftMaxwithLoss, self).__init__()
        self.softmax = nn.LogSoftmax(dim=1)
        self.criterion = nn.NLLLoss(ignore_index=255)

    def forward(self, out, label):
        assert not label.requires_grad
        # out shape  batch_size x channels x h x w
        # label shape batch_size x 1 x h x w
        ## print(out.shape, torch.unique(label))
        label = label[:, 0, :, :].long()
        loss = self.criterion(self.softmax(out), label)

        return loss


class BalancedCrossEntropyLoss(Module):
    """
    Balanced Cross Entropy Loss with optional ignore regions
    """

    def __init__(self, size_average=True, batch_average=True, pos_weight=None):
        super(BalancedCrossEntropyLoss, self).__init__()
        self.size_average = size_average
        self.batch_average = batch_average
        self.pos_weight = pos_weight

    def forward(self, output, label, void_pixels=None):
        #breakpoint()
        assert (output.size() == label.size())
        labels = torch.ge(label, 0.5).float()

        # Weighting of the loss, default is HED-style
        if self.pos_weight is None:
            num_labels_pos = torch.sum(labels)
            num_labels_neg = torch.sum(1.0 - labels)
            num_total = num_labels_pos + num_labels_neg
            w = num_labels_neg / num_total
        else:
            w = self.pos_weight

        output_gt_zero = torch.ge(output, 0).float()
        loss_val = torch.mul(output, (labels - output_gt_zero)) - torch.log(
            1 + torch.exp(output - 2 * torch.mul(output, output_gt_zero)))

        loss_pos_pix = -torch.mul(labels, loss_val)
        loss_neg_pix = -torch.mul(1.0 - labels, loss_val)

        if void_pixels is not None and not self.pos_weight:
            w_void = torch.le(void_pixels, 0.5).float()
            loss_pos_pix = torch.mul(w_void, loss_pos_pix)
            loss_neg_pix = torch.mul(w_void, loss_neg_pix)
            num_total = num_total - torch.ge(void_pixels, 0.5).float().sum()
            w = num_labels_neg / num_total

        loss_pos = torch.sum(loss_pos_pix)
        loss_neg = torch.sum(loss_neg_pix)

        final_loss = w * loss_pos + (1 - w) * loss_neg

        if self.size_average:
            final_loss /= float(np.prod(label.size()))
        elif self.batch_average:
            final_loss /= label.size()[0]

        return final_loss

class BalancedBinaryCrossEntropyLoss(nn.Module):
    """
    Balanced binary cross entropy loss with ignore regions.
    """
    def __init__(self, pos_weight=None, ignore_index=255):
        super().__init__()
        self.pos_weight = pos_weight
        self.ignore_index = ignore_index

    def forward(self, output, label, reduction='mean'):

        mask = (label != self.ignore_index)
        masked_label = torch.masked_select(label, mask)
        masked_output = torch.masked_select(output, mask)

        # weighting of the loss, default is HED-style
        if self.pos_weight is None:
            num_labels_neg = torch.sum(1.0 - masked_label)
            num_total = torch.numel(masked_label)
            w = num_labels_neg / num_total
            if w == 1.0:
                return 0
        else:
            w = torch.as_tensor(self.pos_weight, device=output.device)
        factor = 1. / (1 - w)

        loss = nn.functional.binary_cross_entropy_with_logits(
            masked_output,
            masked_label,
            pos_weight=w*factor,
            reduction=reduction)
        loss /= factor
        return loss
        
class BinaryCrossEntropyLoss(Module):
    """
    Binary Cross Entropy with ignore regions, not balanced.
    """

    def __init__(self, size_average=True, batch_average=True):
        super(BinaryCrossEntropyLoss, self).__init__()
        self.size_average = size_average
        self.batch_average = batch_average

    def forward(self, output, label, void_pixels=None):
        assert (output.size() == label.size())

        labels = torch.ge(label, 0.5).float()

        output_gt_zero = torch.ge(output, 0).float()
        loss_val = torch.mul(output, (labels - output_gt_zero)) - torch.log(
            1 + torch.exp(output - 2 * torch.mul(output, output_gt_zero)))

        loss_pos_pix = -torch.mul(labels, loss_val)
        loss_neg_pix = -torch.mul(1.0 - labels, loss_val)

        if void_pixels is not None:
            w_void = torch.le(void_pixels, 0.5).float()
            loss_pos_pix = torch.mul(w_void, loss_pos_pix)
            loss_neg_pix = torch.mul(w_void, loss_neg_pix)

        loss_pos = torch.sum(loss_pos_pix)
        loss_neg = torch.sum(loss_neg_pix)
        final_loss = loss_pos + loss_neg

        if self.size_average:
            final_loss /= float(np.prod(label.size()))
        elif self.batch_average:
            final_loss /= label.size()[0]

        return final_loss


class DepthLoss(nn.Module):
    """
    Loss for depth prediction. By default L1 loss is used.  
    """
    def __init__(self, loss='l1'):
        super(DepthLoss, self).__init__()
        if loss == 'l1':
            self.loss = nn.L1Loss()
            #self.loss = nn.SmoothL1Loss()

        else:
            raise NotImplementedError('Loss {} currently not supported in DepthLoss'.format(loss))

    def forward(self, out, label):
        ## mask = (label != 255)
        mask1 = (label != 255)
        mask2 = (label < 65535*0.01*0.01)
        mask = mask1.float() * mask2.float()
        mask = mask.bool()

        # print(torch.max(out), torch.max(label))
        return self.loss(torch.masked_select(out, mask), torch.masked_select(label, mask))



class Normalize(nn.Module):
    def __init__(self):
        super(Normalize, self).__init__()

    def forward(self, bottom):
        qn = torch.norm(bottom, p=2, dim=1).unsqueeze(dim=1) + 1e-12
        top = bottom.div(qn)

        return top


class NormalsLoss(Module):
    """
    L1 loss with ignore labels
    normalize: normalization for surface normals
    """
    def __init__(self, size_average=True, normalize=False, norm=2):
        super(NormalsLoss, self).__init__()

        self.size_average = size_average

        if normalize:
            self.normalize = Normalize()
        else:
            self.normalize = None

        if norm == 1:
            print('Using L1 loss for surface normals')
            self.loss_func = F.l1_loss
        elif norm == 2:
            print('Using L2 loss for surface normals')
            self.loss_func = F.mse_loss
        else:
            raise NotImplementedError

    def forward(self, out, label, ignore_label=255):
        assert not label.requires_grad
        mask = (label != ignore_label)
        n_valid = torch.sum(mask).item()

        if self.normalize is not None:
            out_norm = self.normalize(out)
            loss = self.loss_func(torch.masked_select(out_norm, mask), torch.masked_select(label, mask), reduction='sum')
        else:
            loss = self.loss_func(torch.masked_select(out, mask), torch.masked_select(label, mask), reduction='sum')

        if self.size_average:
            if ignore_label:
                ret_loss = torch.div(loss, max(n_valid, 1e-6))
                return ret_loss
            else:
                ret_loss = torch.div(loss, float(np.prod(label.size())))
                return ret_loss

        return loss

    

class ReconstructionLoss(Module):
    """
    L1 loss with ignore labels
    normalize: normalization for surface normals
    """
    def __init__(self, size_average=True, normalize=False, norm=2):
        super(ReconstructionLoss, self).__init__()

        self.size_average = size_average

        if normalize:
            self.normalize = Normalize()
        else:
            self.normalize = None

        if norm == 1:
            print('Using L1 loss for reconstruction')
            self.loss_func = F.l1_loss
        elif norm == 2:
            print('Using L2 loss for reconstruction')
            self.loss_func = F.mse_loss
        else:
            raise NotImplementedError

    def forward(self, out, label, ignore_label=255):
        assert not label.requires_grad
        mask = (label != ignore_label)
        n_valid = torch.sum(mask).item()

        if self.normalize is not None:
            out_norm = self.normalize(out)
            loss = self.loss_func(torch.masked_select(out_norm, mask), torch.masked_select(label, mask), reduction='sum')
        else:
            loss = self.loss_func(torch.masked_select(out, mask), torch.masked_select(label, mask), reduction='sum')

        if self.size_average:
            if ignore_label:
                ret_loss = torch.div(loss, max(n_valid, 1e-6))
                return ret_loss
            else:
                ret_loss = torch.div(loss, float(np.prod(label.size())))
                return ret_loss

        return loss

class L1Loss(nn.Module):
    """
    from ATRC
    L1 loss with ignore regions.
    normalize: normalization for surface normals
    """
    def __init__(self, normalize=False, ignore_index=0, ignore_invalid_area=True):
        super().__init__()
        self.normalize = normalize
        self.ignore_invalid_area = ignore_invalid_area
        if ignore_invalid_area:
            self.ignore_index = ignore_index

    def forward(self, out, label, reduction='mean'):

        if self.normalize:
            out = nn.functional.normalize(out, p=2, dim=1)

        if self.ignore_invalid_area:
            mask = (label != self.ignore_index).all(dim=1, keepdim=True)
        else:
            mask = torch.ones_like(label).all(dim=1, keepdim=True)
        n_valid = torch.sum(mask).item()
        masked_out = torch.masked_select(out, mask)
        masked_label = torch.masked_select(label, mask)
        if reduction == 'mean':
            return nn.functional.l1_loss(masked_out, masked_label, reduction='sum') / max(n_valid, 1)
        elif reduction == 'sum':
            return nn.functional.l1_loss(masked_out, masked_label, reduction='sum')
        elif reduction == 'none':
            return nn.functional.l1_loss(masked_out, masked_label, reduction='none')