from easydict import EasyDict as edict

import os


def get_depth_width(model_name):
    if model_name == "l":
        depth = 1.0
        width = 1.0
    elif model_name == "m":
        depth = 0.67
        width = 0.75
    elif model_name == "s":
        depth = 0.33
        width = 0.50
    elif model_name == "tiny":
        depth = 0.33
        width = 0.375
    else:
        raise ValueError(f"不支持{model_name}")
    return depth, width


def get_config():
    conf = edict()
    conf.save_path = "./checkpoints"

    conf.num_classes = 2
    conf.depth, conf.width = get_depth_width("tiny")

    # ---------------- dataloader config ---------------- #
    # set worker to 4 for shorter dataloader init time
    conf.num_workers = 16
    conf.input_size = (640, 640)  # (height, width)
    conf.multiscale_range = 5

    conf.data_dir = 'D:\\temp_data\\voc\\VOCdevkit'
    # conf.images_path = r"D:\temp_data\mask_face_objdet\train"
    # conf.train_path = "E:/Learn/MaskDetector/anno/train.txt"
    # conf.val_path = "E:/Learn/MaskDetector/anno/val.txt"
    conf.train_image_set = [("2007", "trainval"), ("2012", "trainval")]
    conf.valid_image_set = [("2007", "test")]

    # --------------- transform config ----------------- #
    conf.lr = 1e-3
    conf.mosaic_prob = 1.0
    conf.mixup_prob = 1.0
    conf.hsv_prob = 1.0
    conf.flip_prob = 0.5
    conf.degrees = 10.0
    conf.translate = 0.1
    conf.mosaic_scale = (0.1, 2)
    conf.mixup_scale = (0.5, 1.5)
    conf.shear = 2.0
    conf.perspective = 0.0
    conf.enable_mixup = True
    conf.enable_mosaic = True

    # --------------  training config --------------------- #
    conf.batch_size = 32
    conf.min_face_size = 30
    conf.warmup_epochs = 5
    conf.max_epoch = 400
    conf.warmup_lr = 0
    conf.basic_lr_per_img = 0.01 / 64.0
    conf.scheduler = "yoloxwarmcosyoloxwarmcos"
    conf.no_aug_epochs = 15
    conf.min_lr_ratio = 0.05
    conf.ema = True

    conf.weight_decay = 5e-4
    conf.momentum = 0.9
    conf.print_interval = 10
    conf.eval_interval = 10
    conf.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

    # -----------------  testing config ------------------ #
    conf.test_size = (640, 640)
    conf.test_conf = 0.7
    conf.nmsthre = 0.3
    conf.confthre = 0.6
    return conf

conf = get_config()
