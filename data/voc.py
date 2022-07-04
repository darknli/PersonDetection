#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Code are based on
# https://github.com/fmassa/vision/blob/voc_dataset/torchvision/datasets/voc.py
# Copyright (c) Francisco Massa.
# Copyright (c) Ellis Brown, Max deGroot.
# Copyright (c) Megvii, Inc. and its affiliates.

import os
import os.path
from glob import glob
import random
import xml.etree.ElementTree as ET
from loguru import logger
from torch_frame.vision import xyxy2cxcywh, mixup_boxes, mosaic
import cv2
import numpy as np
from data.expand_body_face_data import BodyFaceData
from detection.utils.datasets_wrapper import Dataset
from .voc_classes import VOC_CLASSES
import traceback


class AnnotationTransform(object):
    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, class_to_ind=None, keep_difficult=True):
        self.class_to_ind = class_to_ind or dict(
            zip(VOC_CLASSES, range(len(VOC_CLASSES)))
        )
        self.keep_difficult = keep_difficult

    def __call__(self, target):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = np.empty((0, 5))
        for obj in target.iter("object"):
            difficult = obj.find("difficult")
            if difficult is not None:
                difficult = int(difficult.text) == 1
            else:
                difficult = False
            if not self.keep_difficult and difficult:
                continue
            name = obj.find("name").text.strip()
            bbox = obj.find("bndbox")

            pts = ["xmin", "ymin", "xmax", "ymax"]
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1
                # scale height or width
                # cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                bndbox.append(cur_pt)
            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            res = np.vstack((res, bndbox))  # [xmin, ymin, xmax, ymax, label_ind]
            # img_id = target.find('filename').text[:-4]

        width = int(target.find("size").find("width").text)
        height = int(target.find("size").find("height").text)
        img_info = (height, width)

        return res, img_info


class VOCDetection(Dataset):
    """
    VOC Detection Dataset Object

    input is image, target is annotation

    Args:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    def __init__(
            self,
            data_dir,
            image_sets=[("2007", "trainval"), ("2012", "trainval")],
            img_size=(416, 416),
            preproc=None,
            target_transform=AnnotationTransform(),
            dataset_name="VOC0712",
            cache=False,
            min_face=20,
            max_boxes=300,
            mixup=False,
            mosaic=False
    ):
        super().__init__(img_size)
        self.root = data_dir
        self.image_set = image_sets
        self.img_size = img_size
        self.preproc = preproc
        self.target_transform = target_transform
        self.name = dataset_name
        self._annopath = os.path.join("%s", "Annotations", "%s.xml")
        self._imgpath = os.path.join("%s", "JPEGImages", "%s.jpg")
        self._classes = VOC_CLASSES
        self.person_cls_idx = VOC_CLASSES.index("person")
        self.ids = list()
        self.max_boxes = max_boxes
        num_skip = 0
        for (year, name) in image_sets:
            self._year = year
            rootpath = os.path.join(self.root, "VOC" + year)
            for line in open(
                    os.path.join(rootpath, "ImageSets", "Main", name + ".txt")
            ):
                target = ET.parse(self._annopath % (rootpath, line.strip())).getroot()
                assert self.target_transform is not None
                res, _ = self.target_transform(target)
                res = res[res[:, -1] == self.person_cls_idx]
                if len(res) == 0 and np.random.random() > 0.2:
                    num_skip += 1
                    continue
                self.ids.append((rootpath, line.strip()))
        print(f"跳过了{num_skip}个样本")
        self.annotations = self._load_coco_annotations()
        self.imgs = None
        self.min_face = min_face
        self.mosaic = mosaic
        self.mixup = mixup
        self.expand_data = []
        if self.name == "train":
            self.expand_data = BodyFaceData(len(self.ids), "D:/temp_data/labelme_bodyface")

        # self.all_images = {img_id: cv2.imread(self._imgpath % img_id) for img_id in self.ids}
        if cache:
            self._cache_images()

    def __len__(self):
        return len(self.ids) + len(self.expand_data)

    def _load_coco_annotations(self):
        return [self.load_anno_from_ids(_ids) for _ids in range(len(self.ids))]

    def _cache_images(self):
        logger.warning(
            "\n********************************************************************************\n"
            "You are using cached images in RAM to accelerate training.\n"
            "This requires large system RAM.\n"
            "Make sure you have 60G+ RAM and 19G available disk space for training VOC.\n"
            "********************************************************************************\n"
        )
        max_h = self.img_size[0]
        max_w = self.img_size[1]
        cache_file = self.root + "/img_resized_cache_" + self.name + ".array"
        if not os.path.exists(cache_file):
            logger.info(
                "Caching images for the frist time. This might take about 3 minutes for VOC"
            )
            self.imgs = np.memmap(
                cache_file,
                shape=(len(self.ids), max_h, max_w, 3),
                dtype=np.uint8,
                mode="w+",
            )
            from tqdm import tqdm
            from multiprocessing.pool import ThreadPool

            NUM_THREADs = min(8, os.cpu_count())
            loaded_images = ThreadPool(NUM_THREADs).imap(
                lambda x: self.load_resized_img(x),
                range(len(self.annotations)),
            )
            pbar = tqdm(enumerate(loaded_images), total=len(self.annotations))
            for k, out in pbar:
                self.imgs[k][: out.shape[0], : out.shape[1], :] = out.copy()
            self.imgs.flush()
            pbar.close()
        else:
            logger.warning(
                "You are using cached imgs! Make sure your dataset is not changed!!"
            )

        logger.info("Loading cached imgs...")
        self.imgs = np.memmap(
            cache_file,
            shape=(len(self.ids), max_h, max_w, 3),
            dtype=np.uint8,
            mode="r+",
        )

    def load_anno_from_ids(self, index):
        img_id = self.ids[index]
        target = ET.parse(self._annopath % img_id).getroot()

        assert self.target_transform is not None
        res, img_info = self.target_transform(target)
        height, width = img_info

        r = min(self.img_size[0] / height, self.img_size[1] / width)
        res[:, :4] *= r
        resized_info = (int(height * r), int(width * r))

        return (res, img_info, resized_info)

    def load_anno(self, index):
        return self.annotations[index][0]

    def load_resized_img(self, index):
        img = self.load_image(index)
        return img

    def load_image(self, index):
        img_id = self.ids[index]
        img = cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR)
        # img = self.all_images[img_id]
        assert img is not None

        return img

    def pull_item(self, index):
        """Returns the original image and target at an index for mixup

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            img, target
        """
        if index >= len(self.ids):
            return self.expand_data.get(index)
        if self.imgs is not None:
            target, img_info, resized_info = self.annotations[index]
            pad_img = self.imgs[index]
            img = pad_img[: resized_info[0], : resized_info[1], :].copy()
        else:
            img = self.load_resized_img(index)
            target, img_info, _ = self.annotations[index]

        # 身体是0，脸是1
        r = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])
        target[:, :-1] = target[:, :-1] / r
        target = target[target[:, -1] == self.person_cls_idx]
        target[:, -1] = 0
        face_boxes = []
        if len(target) > 0:
            with open(f"D:/temp_data/voc/VOCdevkit/faces_anno/{self.ids[index][1]}.jpg") as f:
                for line in f:
                    x1, y1, y2, x2 = np.float32(line.split(","))
                    minx = np.maximum(target[:, 0], x1)
                    miny = np.maximum(target[:, 1], y1)
                    maxx = np.minimum(target[:, 2], x2)
                    maxy = np.minimum(target[:, 3], y2)
                    area = np.maximum(maxx - minx, 0) * np.maximum(maxy - miny, 0)
                    ratio = area / ((x2 - x1) * (y2 - y1))
                    # 防止人脸模型误检
                    if ratio.max() < 0.99:
                        continue
                    face_boxes.append([x1, y1, x2, y2, 1])
        # 防止模型给出的结果不对，用图中身体的标注来限制人脸框
        if len(face_boxes) == 0 or len(target) == 0:
            face_boxes = np.zeros((0, 5))
        else:
            face_boxes = np.float32(face_boxes)

        target = np.concatenate([face_boxes, target], 0)
        target = target[:, (4, 0, 1, 2, 3)]  # 把类别放到前面
        # for box in target:
        #     x1, y1, x2, y2 = box[1:].astype(int)
        #     if box[0] == 0:
        #         c = (0, 255, 0)
        #     else:
        #         c = (255, 0, 0)
        #     cv2.rectangle(img, (x1, y1), (x2, y2), c, 1, cv2.LINE_4)
        # cv2.imshow("image", img)
        # cv2.waitKey()
        return img, target

    @Dataset.mosaic_getitem
    def __getitem__(self, index):
        try:
            if self.mosaic and random.random() < 0.3:
                data = [self.pull_item(index)] + [self.pull_item(random.choice(range(len(self)))) for _ in range(3)]
                image, bboxes = mosaic(data, (640, 640))
            else:
                image, bboxes = self.pull_item(index)
        except:
            print(traceback.format_exc())
            exit()

        if self.mixup and random.random() < 0.5:
            idx2 = random.choice(range(len(self)))
            image2, bboxes2 = self.pull_item(idx2)
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

    def _get_voc_results_file_template(self):
        filename = "comp4_det_test" + "_{:s}.txt"
        filedir = os.path.join(self.root, "results", "VOC" + self._year, "Main")
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        path = os.path.join(filedir, filename)
        return path

    def _write_voc_results_file(self, all_boxes):
        for cls_ind, cls in enumerate(VOC_CLASSES):
            cls_ind = cls_ind
            if cls == "__background__":
                continue
            print("Writing {} VOC results file".format(cls))
            filename = self._get_voc_results_file_template().format(cls)
            with open(filename, "wt") as f:
                for im_ind, index in enumerate(self.ids):
                    index = index[1]
                    dets = all_boxes[cls_ind][im_ind]
                    if dets == []:
                        continue
                    for k in range(dets.shape[0]):
                        f.write(
                            "{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n".format(
                                index,
                                dets[k, -1],
                                dets[k, 0] + 1,
                                dets[k, 1] + 1,
                                dets[k, 2] + 1,
                                dets[k, 3] + 1,
                            )
                        )
