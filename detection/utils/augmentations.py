from random import random, randint, choice
import cv2
import numpy as np


def get_iou(bbox1, bbox2):
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    inter = max((x2 - x1), 0)* max((y2 - y1), 0)
    iou = inter / ((bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1]) +
               (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1]) - inter)
    return iou


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
            boxes[:, 1::2] = width - boxes[:, 1::2]
            # tmp = boxes[:, 1].copy()
            # boxes[:, 1] = boxes[:, 3]
            # boxes[:, 3] = tmp
            boxes[:, 3], boxes[:, 1] = boxes[:, 1].copy(), boxes[:, 3].copy()
            return image, boxes
        else:
            return image


class RandomCrop:
    def __init__(self):
        # self.crop_selector = ["whole", 0.1, 0.3, 0.5, 0.7, "random"]
        self.crop_selector = ["random"]
        self.min_face_crop = {
            0.1: 50 ** 2,
            0.3: 70 ** 2,
            0.5: 90 ** 2,
            0.7: 120 ** 2
        }

    def __call__(self, x):
        image, bboxes = x
        crop_action = choice(self.crop_selector)
        bboxes = bboxes[:, 1:]
        height, width = image.shape[:2]
        # print(crop_action)
        if crop_action == "whole":
            return x
        elif crop_action == "random":
            x1, y1, x2, y2 = self._random_crop(width, height)
        else:
            np.random.shuffle(bboxes)
            has_choose = False
            for bbox in bboxes.copy():
                area = (bbox[3] - bbox[1]) * (bbox[2] - bbox[0])
                if area < self.min_face_crop[crop_action]:
                    continue
                while True:
                    ratio = (1 / crop_action) ** 0.5
                    bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
                    if bw <= 0 or bh <= 0:
                        continue
                    w = randint(int(bw / ratio), int(bw * ratio))
                    min_h, max_h = max(int(bh / ratio), int(w * 0.5)), min(int(bh * ratio), int(w * 2))
                    if min_h > max_h:
                        break
                    h = randint(min_h, max_h + 1)
                    x1 = randint(0, max(bbox[2] - w, 1))
                    y1 = randint(0, max(bbox[3] - h, 1))
                    x2 = min(x1 + w, width)
                    y2 = min(y1 + h, height)
                    if get_iou(np.array([x1, y1, x2, y2]), bbox) > crop_action:
                        has_choose = True
                        break
                if has_choose:
                    break
            # 如果都没选，则直接随机裁剪
            if not has_choose:
                if random() > 0.5:
                    return x
                else:
                    x1, y1, x2, y2 = self._random_crop(width, height)

        image = image[y1: y2, x1: x2, :]
        crop_boxes = []

        # tmp = image.copy()
        for bbox in bboxes:
            xcenter = (bbox[0] + bbox[2]) / 2
            ycenter = (bbox[1] + bbox[3]) / 2
            if x1 <= xcenter <= x2 and y1 <= ycenter <= y2:
                crop_boxes.append([0, bbox[0]-x1, bbox[1]-y1, bbox[2]-x1, bbox[3]-y1])
        #         cv2.rectangle(tmp, (int(bbox[0]-x1), int(bbox[1]-y1)), (int(bbox[2]-x1), int(bbox[3]-y1)), (0, 255, 0), 2, cv2.LINE_4)
        # print("crop", crop_action, len(bboxes), len(crop_boxes), [x2 - x1, y2 - y1])

        # cv2.imshow("t", tmp)
        # cv2.waitKey()
        # cv2.destroyWindow("t")

        return image, np.array(crop_boxes)

    def _random_crop(self, width, height):
        if width < height:
            w = randint(int(width * 0.3), width)
            h = min(randint(int(w * 0.5), w * 2), height)
        else:
            h = randint(int(height * 0.3), height)
            w = min(randint(int(h * 0.5), h * 2), width)
        x1 = randint(0, width - w)
        y1 = randint(0, height - h)
        x2 = x1 + w
        y2 = y1 + h
        return x1, y1, x2, y2


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
            # print(image.shape)
            # for box in boxes:
            #     print("box", box[1:].astype(int))
            #     x1, y1, x2, y2 = box[1:].astype(int)
            #     cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 1, cv2.LINE_4)
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

