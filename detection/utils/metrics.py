import copy

import numpy as np
import torch
from .boxes import matrix_iou


def cal_match(pred_box, gt_box):
    count = 0
    iou = matrix_iou(pred_box[:, 1:], gt_box[:, 1:])
    for i, pbox in enumerate(pred_box):
        for j, gbox in enumerate(copy.deepcopy(gt_box)):
            if iou[i][j] >= 0.9 and np.abs(pbox[1:] - gbox[1:]).max() < 5 and pbox[0] == gbox[0]:
                count += 1
                gt_box = np.concatenate([gt_box[:j], gt_box[j+1:]])
                iou = np.concatenate([iou[:, :j], iou[:, j+1:]], 1)
                break
    return count / len(pred_box)
