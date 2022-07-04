#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import torch
import torch.nn as nn
from itertools import combinations


class IOUloss(nn.Module):
    def __init__(self, reduction="none", loss_type="iou"):
        super(IOUloss, self).__init__()
        self.reduction = reduction
        self.loss_type = loss_type

    def forward(self, pred, target):
        assert pred.shape[0] == target.shape[0]

        pred = pred.view(-1, 4)
        target = target.view(-1, 4)
        tl = torch.max(
            (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
        )
        br = torch.min(
            (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
        )

        area_p = torch.prod(pred[:, 2:], 1)
        area_g = torch.prod(target[:, 2:], 1)

        en = (tl < br).type(tl.type()).prod(dim=1)
        area_i = torch.prod(br - tl, 1) * en
        iou = (area_i) / (area_p + area_g - area_i + 1e-16)

        if self.loss_type == "iou":
            loss = 1 - iou ** 2
        elif self.loss_type == "giou":
            c_tl = torch.min(
                (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
            )
            c_br = torch.max(
                (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
            )
            area_c = torch.prod(c_br - c_tl, 1)
            giou = iou - (area_c - area_i) / area_c.clamp(1e-16)
            loss = 1 - giou.clamp(min=-1.0, max=1.0)

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss


def quadrilateral_grad_loss(pred_y, true_y, mask):
    """
    保证pred和gt的四边形是相似的(数学意义上)
    :param mat:
    :param pred:
    :return:
    """

    tot_mae_loss = 0
    # weights = torch.ones_like(pred_y, dtype=pred_y.dtype, device=pred_y.device)
    # weights[..., 0] = 960
    # weights[..., 1] = 512
    # pred_y = pred_y * weights
    # true_y = true_y * weights

    for i, j in combinations(range(4), 2):
        mask_ij = mask[:, i] & mask[:, j]

        pred_vec_x = pred_y[:, i, 0] - pred_y[:, j, 0]
        pred_vec_y = pred_y[:, i, 1] - pred_y[:, j, 1]

        true_vec_x = true_y[:, i, 0] - true_y[:, j, 0]
        true_vec_y = true_y[:, i, 1] - true_y[:, j, 1]

        cosine_sim = (pred_vec_x * true_vec_x + pred_vec_y * true_vec_y) / \
                     (torch.sqrt(pred_vec_x**2+pred_vec_y**2) * torch.sqrt(true_vec_x**2+true_vec_y**2))

        cosine_dist = 1 - cosine_sim
        euc_dist = torch.abs(torch.sqrt(pred_vec_x**2+pred_vec_y**2) - torch.sqrt(true_vec_x**2+true_vec_y**2))

        # 斜率的MAE不稳定, 弃用
        # # pred斜率
        # pred_k = pred_vec_x / pred_vec_y
        # # true斜率
        # true_k = true_vec_x / true_vec_y
        # mae = torch.abs(pred_k - true_k)
        # mae = mae[mask_ij]

        cosine_dist = cosine_dist[mask_ij]
        if len(cosine_dist) > 0:
            tot_mae_loss += cosine_dist.mean()
            tot_mae_loss += euc_dist[mask_ij].mean()
            # count += len(cosine_dist)
    # tot_mae_loss = torch.max(torch.stack(tot_mae_loss, 0), 0)[0].sum()
    # tot_mae_loss = torch.cat(tot_mae_loss, 0).mean()
    # tot_mae_loss = tot_mae_loss / len()
    return tot_mae_loss

