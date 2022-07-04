import os.path
from scipy.io import loadmat
from torch.utils.data import Dataset
from os.path import join, exists
from glob import glob
from .boxes import xyxy2cxcywh
from tqdm import tqdm
import cv2
import torch
import numpy as np
import json
from random import random, randint


class WiderDataset(Dataset):
    def __init__(self, root, file, transform=None, max_boxes=300, min_face=32, cache=True):
        super(WiderDataset, self).__init__()
        self.transform = transform
        self.num_classes = 1
        self.max_boxes = max_boxes
        self.min_face = min_face
        self.anno = self._decode_mat(root, file, cache)
        print("读取数据成功")

    def _get_anno(self, root, file, cache):
        anno = []
        item = None

        with open(file) as f:
            for line in f:
                line = line.strip()
                if line.endswith("jpg"):
                    if item is not None:
                        item["boxes"] = np.array(item["boxes"], dtype=int)
                        anno.append(item)
                    item = {
                        "filename": os.path.join(root, line),
                        "boxes": []
                    }
                else:
                    box = line.split(" ")
                    if len(box) == 1:
                        continue
                    box = box[:4]
                    item["boxes"].append([0] + box)
        item["boxes"] = np.array(item["boxes"], dtype=int)
        anno.append(item)
        return anno

    def _decode_mat(self, root, file, cache):
        anno = []
        data = loadmat(file)
        event_list = data["event_list"]
        file_list = data["file_list"]
        face_bbx_list = data['face_bbx_list']
        occlusion_label_list = data['occlusion_label_list']
        for event_idx, event in enumerate(event_list):
            directory = event[0][0]

            for file, bbx, occlusion in zip(file_list[event_idx][0],
                                 face_bbx_list[event_idx][0],
                                 occlusion_label_list[event_idx
                                            ][0]):
                f = file[0][0]
                path_of_image = os.path.join(root, directory, f + '.jpg')

                bboxes = []
                bbx0 = bbx[0]
                occlusion_labels = []
                occlusion = occlusion[0]
                flag = False
                for i in range(bbx0.shape[0]):
                    occlusion_labels.append(occlusion[i])
                    xmin, ymin, xoffset, yoffset = bbx0[i]
                    xmax = xmin + xoffset
                    ymax = ymin + yoffset
                    if xmax < xmin or ymax < ymin:
                        flag = True
                        break
                    bboxes.append((0, int(xmin), int(ymin), int(xmax), int(ymax)))
                if flag:
                    continue
                bboxes = np.array(bboxes, dtype=int)
                area = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 4] - bboxes[:, 2])
                if np.all(area < 60 **2):
                    # print(f"脸太小，跳过{path_of_image}")
                    continue

                item = {
                    "filename": path_of_image,
                    "boxes": bboxes
                }
                anno.append(item)
        return anno

    def __getitem__(self, idx):
        item = self.anno[idx]
        filename = item["filename"]
        if isinstance(filename, str):
            image = cv2.imread(filename)
        else:
            image = filename

        bboxes = item["boxes"].astype(np.float32)
        x = image, bboxes

        if self.transform is not None:
            image, bboxes = self.transform(x)

        if len(bboxes) == 0:
            bboxes = np.zeros((self.max_boxes, 5))
        else:
            area = np.maximum(bboxes[:, 3] - bboxes[:, 1], 0) * np.maximum(bboxes[:, 4] - bboxes[:, 2], 0)
            new_bboxes = np.zeros((self.max_boxes, 5))
            bboxes = bboxes[area >= (self.min_face**2)]
            new_bboxes[:len(bboxes)] = bboxes
            bboxes = new_bboxes

        bboxes[:, 1:] = xyxy2cxcywh(bboxes[:, 1:])
        bboxes = np.ascontiguousarray(bboxes)
        return image, bboxes

    def __len__(self):
        return len(self.anno)
