from models import *
from torch import nn


def get_model(config):
    def init_yolo(M):
        for m in M.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-3
                m.momentum = 0.03

    in_channels = [256, 512, 1024]
    backbone = YOLOPAFPN(config.depth, config.width, in_channels=in_channels)
    head = YOLOXHead(config.num_classes, config.width, in_channels=in_channels)
    model = YOLOX(backbone, head)

    model.apply(init_yolo)
    model.head.initialize_biases(1e-2)
    return model