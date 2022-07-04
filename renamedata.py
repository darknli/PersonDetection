import random
from typing import Union
from torch.utils.data.dataloader import Dataset
import numpy as np
import cv2
from torch_frame.vision import xyxy2cxcywh, mixup_boxes, mosaic
from glob import glob
import os
import json


class PrivatesData(Dataset):
    def __init__(self, root, transform, max_boxes=300, min_size_obj=10, exp_root=None, mixup=False, mosaic=False):
        super(PrivatesData, self).__init__()
        self.transform = transform
        self.max_boxes = max_boxes
        self.min_size_obj = min_size_obj
        self.anno = self._decode_expand_data(root)
        if exp_root is not None:
            self.anno += self._decode_expand_data(exp_root)
        self.mixup = mixup
        self.mosaic = mosaic

    def _decode_expand_data(self, root):
        anno_list = []
        for directory in glob(os.path.join(root, "*")):
            anno = self._get_anno(directory)
            anno_list += anno
        return anno_list

    @ staticmethod
    def _get_anno(root):
        files = glob(os.path.join(root, "*.json"))
        anno = []
        for file in files:
            with open(file, encoding="utf-8") as f:
                text = json.loads(f.read())
            boxes = []
            for item in text["shapes"]:
                # label=4的话变成3，因为压根没有3
                b = [int(item["label"]) if item["label"] != "4" else 3] + item["points"][0] + item["points"][1]
                boxes.append(b)

            image_path = file.replace("json", "jpg")
            if not os.path.exists(image_path):
                continue
            anno.append(
                {
                    "filename": image_path,
                    "boxes": np.array(boxes)
                }
            )
        return anno

    def __len__(self):
        return len(self.anno)

    def get_one_example(self, idx):
        item = self.anno[idx]
        filename = item["filename"]
        if isinstance(filename, str):
            image = cv2.imread(filename)
        else:
            image = filename
        bboxes = item["boxes"].astype(np.float32)
        return image, bboxes

    def __getitem__(self, idx):
        if self.mosaic and random.random() < 0.3:
            data = [self.get_one_example(idx)] + [self.get_one_example(random.choice(range(len(self.anno)))) for _ in range(3)]
            image, bboxes = mosaic(data, (560, 560))
        else:
            image, bboxes = self.get_one_example(idx)
        # print(bboxes.shape)
        # cv2.imshow("1", image)
        # cv2.waitKey()
        if self.mixup and random.random() < 0.5:
            idx2 = random.choice(range(len(self.anno)))
            image2, bboxes2 = self.get_one_example(idx2)
            image, bboxes = mixup_boxes(image, bboxes, image2, bboxes2, wh_thr=0)
        # for box
        # if len(bboxes) > 0 and bboxes[:, 0].max() >= 4:
        #     print("debug")
        # for b in bboxes:
        #     cv2.rectangle(image, tuple(b[1:3].astype(int)), tuple(b[3:].astype(int)), (0, 255, 0), 1)
        # cv2.imshow("1", image)
        # cv2.waitKey()
        x = image, bboxes

        if self.transform is not None:
            image, bboxes = self.transform(x)

        if len(bboxes) == 0:
            bboxes = np.zeros((self.max_boxes, 5))
        else:
            area = np.maximum(bboxes[:, 3] - bboxes[:, 1], 0) * np.maximum(bboxes[:, 4] - bboxes[:, 2], 0)
            new_bboxes = np.zeros((self.max_boxes, 5))
            bboxes = bboxes[area >= (self.min_size_obj**2)]
            new_bboxes[:len(bboxes)] = bboxes
            bboxes = new_bboxes

        bboxes[:, 1:] = xyxy2cxcywh(bboxes[:, 1:])
        bboxes = np.ascontiguousarray(bboxes)
        return image, bboxes
