from torch.utils.data.dataset import Dataset
from os.path import join
import cv2
import numpy as np
import random
import math
import torch
from torch_frame.vision import xyxy2cxcywh, mixup_boxes, mosaic


class MaskData(Dataset):
    def __init__(self, data_root, anno_file, input_size, mosaic=False, mixup=False, min_face=20, preproc=None):
        self.anno = self._get_anno(anno_file, data_root)
        self.preproc = preproc
        self.output_size = (int(input_size[0] / 4), int(input_size[1] / 4))
        self.num_classes = 2
        self.mosaic = mosaic
        self.mixup = mixup
        self.max_boxes = 300
        self.min_face = min_face

    def _get_anno(self, anno_file, root):
        anno = []
        with open(anno_file) as f:
            for line in f.readlines():
                item = {}
                fields = line.strip().split(" ")
                name, boxes = fields[0], fields[1:]
                boxes = np.array(boxes).reshape(-1, 6).astype(float)
                boxes = np.ascontiguousarray(np.delete(boxes, 4, 1)[:, (4, 0, 1, 2, 3)])
                item["name"] = name
                item["path"] = join(root, name)
                # if(len(boxes) == 0):
                    # print(name)
                item["boxes_info"] = boxes
                anno.append(item)
        return anno

    def __len__(self):
        return len(self.anno)

    def __getitem__(self, idx):
        # try:
        if self.mosaic and random.random() < 0.3:
            data = [(cv2.imread(self.anno[idx]["path"]), self.anno[idx]["boxes_info"])]
            for _ in range(3):
                item = random.choice(self.anno)
                image = cv2.imread(item["path"])
                data.append((image, item["boxes_info"]))
            image, bboxes = mosaic(data, (640, 640))
        else:
            image, bboxes = cv2.imread(self.anno[idx]["path"]), self.anno[idx]["boxes_info"]
        # except:
        #     print("debug")

        if self.mixup and random.random() < 0.5:
            item = random.choice(self.anno)
            image2, bboxes2 = cv2.imread(item["path"]), item["boxes_info"]
            image, bboxes = mixup_boxes(image, bboxes, image2, bboxes2, wh_thr=0)

        # if self.preproc is not None:
        #     img, target = self.preproc(img, target, self.input_dim)

        bboxes = bboxes.astype(np.float32)
        x = image, bboxes
        if self.preproc is not None:
            image, bboxes = self.preproc(x)

        if len(bboxes) == 0:
            bboxes = np.zeros((self.max_boxes, 5))
        else:
            area = np.maximum(bboxes[:, 3] - bboxes[:, 1], 0) * np.maximum(bboxes[:, 4] - bboxes[:, 2], 0)
            new_bboxes = np.zeros((self.max_boxes, 5))
            bboxes = bboxes[area >= (self.min_face ** 2)]
            new_bboxes[:len(bboxes)] = bboxes
            bboxes = new_bboxes

        bboxes[:, 1:] = xyxy2cxcywh(bboxes[:, 1:])
        bboxes = np.ascontiguousarray(bboxes)
        return image, bboxes
