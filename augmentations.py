from random import random, randint, choice, uniform
from typing import Union

import cv2
import numpy as np
from torch_frame.vision import RandomBrightness, RandomGammaCorrection, RandomHueSaturation, RandomContrast
import logging


def get_iou(bbox1, bbox2):
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    inter = max((x2 - x1), 0) * max((y2 - y1), 0)
    iou = inter / ((bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1]) +
                   (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1]) - inter)
    return iou


class Color:
    def __init__(self):
        self.bri = RandomBrightness()
        self.gamma = RandomGammaCorrection()
        # self.hs = RandomHueSaturation(-6, 6, 0.8, 1.2)
        self.hs = None
        self.constarast = RandomContrast()

    def __call__(self, x):
        if isinstance(x, tuple):
            image, boxes = x
        else:
            image = x
            boxes = None
        if self.bri is not None:
            image = self.bri(image)
        if self.gamma is not None:
            image = self.gamma(image)
        if self.hs is not None:
            image = self.hs(image)
        if self.constarast is not None:
            image = self.constarast(image)
        if boxes is None:
            return image
        else:
            return image, boxes


class Mirror:
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, x):
        if random() > self.prob:
            return x
        if isinstance(x, tuple):
            image, boxes = x
        else:
            image = x
        image = image[:, ::-1, :]
        if isinstance(x, tuple):
            width = image.shape[1]
            if len(boxes) > 0:
                boxes[:, 1::2] = width - boxes[:, 1::2]
                # tmp = boxes[:, 1].copy()
                # boxes[:, 1] = boxes[:, 3]
                # boxes[:, 3] = tmp
                boxes[:, 3], boxes[:, 1] = boxes[:, 1].copy(), boxes[:, 3].copy()
            return image, boxes
        else:
            return image


class RandomCrop:
    def __init__(self, min_pic_size=400):
        self.crop_selector = ["whole", 0.1, 0.3, 0.5, 0.7, "random"]
        self.p = [0.1, 0.2, 0.2, 0.15, 0.15, 0.2]
        # self.crop_selector = ["random"]
        self.min_face_crop = {
            0.1: 200 ** 2,
            0.3: 250 ** 2,
            0.5: 300 ** 2,
            0.7: 400 ** 2
        }
        self.min_pic_size = min_pic_size

    def __call__(self, x):
        image, bboxes = x
        crop_action = np.random.choice(self.crop_selector, 1, replace=False, p=self.p)[0]
        height, width = image.shape[:2]
        # print(crop_action)
        has_choose = False
        if crop_action == "whole":
            return x
        elif crop_action == "random" or min(height, width) <= self.min_pic_size:
            x1, y1, x2, y2 = self._random_crop(width, height)
        else:
            crop_action = float(crop_action)
            np.random.shuffle(bboxes)
            if len(bboxes) > 0:
                for bbox in bboxes[:, 1:].astype(int):
                    area = (bbox[3] - bbox[1]) * (bbox[2] - bbox[0])
                    if area < 40 ** 2:
                        # print(f"面积太小{area}")
                        continue

                    bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
                    if bw <= 0 or bh <= 0:
                        continue
                    ratio = (1 / crop_action) ** 0.5
                    size = int(max(bw, bh) * ratio)
                    if size <= self.min_pic_size:
                        size = int(size * 1.2)
                    if size <= self.min_pic_size:
                        # print(f"size太小={size}")
                        break
                    # 循环个30次必须找到了
                    for _ in range(100):
                        # w = randint(int(bw / ratio), int(bw * ratio))
                        # min_h, max_h = max(int(bh / ratio), int(w * 0.5)), min(int(bh * ratio), int(w * 2))
                        # if min_h > max_h:
                        #     break
                        # h = randint(min_h, max_h + 1)

                        try:
                            w = np.random.randint(self.min_pic_size, min(width, size))
                            h = np.random.randint(self.min_pic_size, min(height, size))
                        except:
                            print(width, height, size, self.min_pic_size)
                        x1 = randint(0, max(bbox[2] - w // 2, 1))
                        y1 = randint(0, max(bbox[3] - h // 2, 1))
                        x2 = min(x1 + w, width)
                        y2 = min(y1 + h, height)
                        if get_iou(np.array([x1, y1, x2, y2]), bbox) > crop_action:
                            # print("iou符合标准")
                            has_choose = True
                            break
                        # else:
                        #     print(f"size={size}, iou={get_iou(np.array([x1, y1, x2, y2]), bbox)}不达标")

                    if has_choose:
                        break
            # 如果都没选，则直接随机裁剪
            if not has_choose:
                # print("选择box失败")
                if random() > 0.5:
                    return x
                else:
                    x1, y1, x2, y2 = self._random_crop(width, height)
            # else:
            #     print("选择box成功")

        image = image[y1: y2, x1: x2, :]
        new_h, new_w = image.shape[:2]
        crop_boxes = []

        # tmp = image.copy()
        for bbox in bboxes:
            xcenter = (bbox[1] + bbox[3]) / 2
            ycenter = (bbox[2] + bbox[4]) / 2
            w = np.clip(bbox[3], x1, x2) - np.clip(bbox[1], x1, x2)
            h = np.clip(bbox[4], y1, y2) - np.clip(bbox[2], y1, y2)
            iof = max(w, 0) * max(h, 0) / ((-bbox[1] + bbox[3]) * (-bbox[2] + bbox[4]))

            # if x1 <= xcenter <= x2 and y1 <= ycenter <= y2:
            if iof > 0.2:
                crop_boxes.append([bbox[0],
                                   max(bbox[1] - x1, 0), max(bbox[2] - y1, 0),
                                   min(bbox[3] - x1, new_w), min(bbox[4] - y1, new_h)
                                   ])
                # cv2.rectangle(tmp, (int(bbox[1] - x1), int(bbox[2] - y1)), (int(bbox[3] - x1), int(bbox[4] - y1)),
                #               (0, 255, 0), 2, cv2.LINE_4)
        # print("crop", has_choose, crop_action, len(bboxes), len(crop_boxes), [x2 - x1, y2 - y1], [height, width])
        #
        # cv2.imshow("t", tmp)
        # cv2.waitKey()
        # cv2.destroyWindow("t")

        return image, np.array(crop_boxes)

    def _random_crop(self, width, height):
        if width < height:
            w = randint(int(width * 0.9), width)
            h = randint(int(w * 0.7), min(w * 2, height))
        else:
            h = randint(int(height * 0.9), height)
            w = randint(int(h * 0.7), min(h * 2, width))
        x1 = randint(0, width - w)
        y1 = randint(0, height - h)
        x2 = x1 + w
        y2 = y1 + h
        return x1, y1, x2, y2

    def cal_psnr(self, std, aim):
        gray = 255
        num = 0
        for i in range(std.shape[0]):
            for j in range(std.shape[1]):
                num += (std[i, j] - aim[i, j]) * (std[i, j] - aim[i, j])
        mse = num / (std.shape[0] * std.shape[1])
        psnr = 10 * np.log(gray * gray / mse, 10)
        return psnr


class Resize:
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, x):
        if isinstance(x, tuple):
            image, boxes = x
        else:
            image = x
        origin_height, origin_width = image.shape[:2]
        image = cv2.resize(image, (self.width, self.height))
        if isinstance(x, tuple):
            if len(boxes) > 0:
                boxes[:, 1::2] *= (self.width / origin_width)
                boxes[:, 2::2] *= (self.height / origin_height)
                mask = np.logical_and(boxes[:, 3] - boxes[:, 1] > 12, boxes[:, 4] - boxes[:, 2] > 12)
                boxes = boxes[mask]
            return image, boxes
        else:
            return image


class Normalization:
    def __init__(self, mean, var):
        self.mean = mean
        self.var = var

    def __call__(self, x):
        if isinstance(x, tuple):
            image, boxes = x
            # min_area = float("inf")
            # wh = None
            # for box in boxes:
            #     x1, y1, x2, y2 = box[1:].astype(int)
            #     # if (x2 - x1) * (y2 - y1) < 20 ** 2:
            #     #     continue
            #     area = (x2 - x1) * (y2 - y1)
            #     if min_area > area:
            #         wh = (x2 - x1), (y2 - y1)
            #         min_area = min(min_area, area)
            #     cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1, cv2.LINE_4)
            # print(f"最小面积={min_area}, wh={wh}")
            # cv2.imshow("image", image)
            # cv2.waitKey()
        else:
            image = x

        image = image.astype("float32")
        image[:, :] -= self.mean
        image[:, :] /= self.var
        if len(image.shape) == 2:
            image = image[:, :, None]
        image = image.transpose((2, 0, 1))
        if isinstance(x, tuple):
            return image, boxes
        else:
            return image



def box_candidates(box1, box2, wh_thr=2, ar_thr=20, area_thr=0.2):
    # box1(4,n), box2(4,n)
    # Compute candidate boxes which include follwing 5 things:
    # box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + 1e-16), h2 / (w2 + 1e-16))  # aspect ratio
    return (
            (w2 > wh_thr)
            & (h2 > wh_thr)
            & (w2 * h2 / (w1 * h1 + 1e-16) > area_thr)
            & (ar < ar_thr)
    )  # candidates


