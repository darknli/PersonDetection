#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
import numpy as np
import cv2
import torch
import torch.nn as nn

from .yolo_head import YOLOXHead
from .yolo_pafpn import YOLOPAFPN
from torch_frame.vision import det_postprocess

class YOLOX(nn.Module):
    """
    YOLOX model module. The module list is defined by create_yolov3_modules function.
    The network returns loss values from three YOLO layers during training
    and detection results during test.
    """

    def __init__(self, backbone=None, head=None):
        super().__init__()
        if backbone is None:
            backbone = YOLOPAFPN()
        if head is None:
            head = YOLOXHead(80)

        self.backbone = backbone
        self.head = head

        self.device = "cpu"

    def forward(self, inputs):
        if isinstance(inputs, tuple) or isinstance(inputs, list):
            if len(inputs) == 2:
                x, targets = inputs
            elif len(inputs) == 1:
                x = inputs[0]
                targets = None
            else:
                raise ValueError("输入数量不对")
        else:
            x = inputs
            targets = None
        x = x.to(self.device)
        # fpn output content features of [dark3, dark4, dark5]

        # np.save(r"D:\code\RemoveMosaic\train.npy", x[:1].cpu().numpy())
        # exit()
        fpn_outs = self.backbone(x)

        if targets is not None:
            targets = targets.to(self.device)

            iou_loss, conf_loss, cls_loss, l1_loss = self.head(
                fpn_outs, targets, x
            )
            outputs = {
                # "total_loss": loss,
                "iou_loss": iou_loss,
                "l1_loss": l1_loss,
                "conf_loss": conf_loss,
                "cls_loss": cls_loss,
                # "num_fg": num_fg,
            }
        else:
            outputs = self.head(fpn_outs)

        # for k, v in outputs.items():
        #     if not isinstance(v, float):
        #         v = v.detach().cpu().item()
        #     print(f"{k}, {v}")
        # print()
        # self.head.training = False
        # with torch.no_grad():
        #     outputs_ = det_postprocess(
        #         self.head(fpn_outs), 4, 0.3,
        #         0.05, class_agnostic=True
        #     )
        # self.head.training = True
        # for i in range(len(x)):
        #     color_dict = {
        #         0: (0, 255, 0),
        #         1: (255, 0, 0),
        #         2: (0, 0, 255),
        #         3: (128, 128, 128)
        #     }
        #     image = x[i].cpu().numpy()
        #     image = np.clip((image * 0.5 + 0.5) * 255, 0, 255).astype(np.uint8).transpose((1, 2, 0))
        #     # origin_h, origin_w = image.shape
        #     if outputs_[i] is None:
        #         outputs_[i] = []
        #     print(image.shape)
        #     image = np.ascontiguousarray(image)
        #     for box in outputs_[i]:
        #         box = box.cpu().numpy()
        #         box, obj_copnf, cls_score, cls = box[:4], box[4], box[5], box[6]
        #         x1 = int(box[0])
        #         x2 = int(box[2])
        #         y1 = int(box[1])
        #         y2 = int(box[3])
        #         # x1, y1, x2, y2 = box.astype(int)
        #         cv2.rectangle(image, (x1, y1), (x2, y2), color_dict[cls], 1, cv2.LINE_4)
        #         cv2.putText(image, "{:.3f}".format(obj_copnf * cls_score), (x1, y1), cv2.FONT_HERSHEY_PLAIN, 4,
        #                     color_dict[cls])
        #     # image = cv2.resize(image, None, fx=0.5, fy=0.5)
        #     image = np.ascontiguousarray(image)
        #     cv2.imshow("image", image)
        #     cv2.waitKey()

        return outputs

    def set_device(self, device):
        self.device = device
        self.to(device)
