import os

import numpy as np
import json
import augmentations
from torchvision.transforms import transforms
import torch
from get_model import get_model
from config import get_config
import argparse
from glob import glob
from torch_frame.vision import det_postprocess
import cv2
import re

config = get_config()


def detect_video(model, dst=None):
    transform = transforms.Compose([
        augmentations.Resize(*config.test_size),
        augmentations.Normalization(128, 128)
    ])

    color_dict = {
        0: (0, 255, 0),
        1: (255, 0, 0),
        2: (0, 0, 255),
        3: (128, 128, 128)
    }
    cap = cv2.VideoCapture(0)
    while True:
        ret, image = cap.read()
        if not ret:
            break
        image = np.ascontiguousarray(image[:, ::-1])
        cv2.imwrite("test.png", image)
        h, w = image.shape[:2]
        max_size = max(h, w)
        new_image = np.full((max_size, max_size, 3), 128, dtype=np.uint8)
        new_image[:h, :w] = image
        image = new_image
        origin_h, origin_w = image.shape[:2]
        x = transform(image.copy())
        x = torch.tensor(x).unsqueeze(0)
        with torch.no_grad():
            outputs = model(x)
            outputs = det_postprocess(
                outputs, config.num_classes, config.confthre,
                config.nmsthre, class_agnostic=True
            )
        if outputs[0] is None:
            outputs[0] = []
        print(image.shape)
        image = np.ascontiguousarray(image)
        for box in outputs[0]:
            box = box.cpu().numpy()
            box, obj_copnf, cls_score, cls = box[:4], box[4], box[5], box[6]
            x1 = int(box[0] * origin_w / config.test_size[1])
            x2 = int(box[2] * origin_w / config.test_size[1])
            y1 = int(box[1] * origin_h / config.test_size[0])
            y2 = int(box[3] * origin_h / config.test_size[0])
            # x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(image, (x1, y1), (x2, y2), color_dict[cls], 1, cv2.LINE_4)
            cv2.putText(image, "{:.3f}".format(obj_copnf*cls_score), (x1, y1), cv2.FONT_HERSHEY_PLAIN, 4,
                        color_dict[cls])
        # image = cv2.resize(image, None, fx=0.5, fy=0.5)
        image = np.ascontiguousarray(image)
        cv2.imshow("image", image)
        cv2.waitKey(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("od train parser")
    parser.add_argument(
        "-w", "--weight", type=str, default="saver/body_s_logs_2022-05-06 20_07_10/checkpoints/best.pth", help="model name"
    )
    args = parser.parse_args()

    weights_path = args.weight

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    model = get_model(config)
    model.set_device(device)
    model.eval()

    weights = torch.load(weights_path)
    model.load_state_dict(weights)
    # model.load_state_dict(weights)
    # dst = re.findall(r"checkpoint/(.+?)/", weights_path)[0]
    dst = "111"
    detect_video(model)