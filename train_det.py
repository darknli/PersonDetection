import torch.cuda
from torch_frame import Trainer, CheckpointerHook, LoggerHook, EvalHook
from get_model import get_model
from torch.optim.adam import Adam
from torchvision.transforms import transforms
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data.dataloader import DataLoader
from data import VOCDetection, MaskData
from config import conf
import augmentations
from torch.cuda.amp import GradScaler, autocast


def eval(model, batch):

    #####################
    # 2. 计算loss #
    #####################
    loss_dict = model(batch)
    if isinstance(loss_dict, torch.Tensor):
        loss_dict = {"total_loss": loss_dict}
    else:
        loss_dict["total_loss"] = sum(loss_dict.values())
    for k, v in loss_dict.items():
        loss_dict[k] = [v.item() if isinstance(v, torch.Tensor) else v]
    return loss_dict


def gen_trainer(conf):
    model = get_model(conf)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.set_device(device)
    optimizer = Adam(model.parameters(), conf.lr)
    lr_scheduler = MultiStepLR(optimizer, [500, 1000, 1500], 0.2)
    train_trans = transforms.Compose([
        # augmentations.RandomCrop(),
        augmentations.Mirror(),
        augmentations.Color(),
        augmentations.Resize(*conf.input_size),
        augmentations.Normalization(128, 128)
    ])
    # train_dataset = MaskData(conf.images_path, conf.train_path,
    #                              conf.input_size, preproc=train_trans, min_face=20, mixup=conf.enable_mixup,
    #                              mosaic=conf.enable_mosaic)
    train_dataset = VOCDetection(conf.data_dir, conf.train_image_set, (640, 640), train_trans, dataset_name='train',
                                 mixup=True, mosaic=True)
    train_dataloader = DataLoader(train_dataset, conf.batch_size, shuffle=True,
                                  num_workers=conf.num_workers, pin_memory=True)

    valid_trans = transforms.Compose([
        augmentations.Resize(*conf.input_size),
        augmentations.Normalization(128, 128)
    ])
    # valid_dataset = MaskData(conf.images_path, conf.train_path,
    #                              conf.input_size, preproc=valid_trans, min_face=20, mixup=False, mosaic=False)
    valid_dataset = VOCDetection(conf.data_dir, conf.valid_image_set, (640, 640), valid_trans, dataset_name='valid',
                                 mixup=False, mosaic=False)
    valid_loader = DataLoader(valid_dataset, conf.batch_size, shuffle=False,
                              num_workers=conf.num_workers, pin_memory=True)
    hooks = [
        EvalHook(valid_loader, eval, save_metric="total_loss", max_first=False, prefix="valid"),
        LoggerHook(modes=["valid"]),
        ]
    trainer = Trainer(model, optimizer, lr_scheduler, train_dataloader,
                      conf.max_epoch, "saver/body_s_logs", hooks=hooks)

    state = torch.load("face_body_yolox_s_models/checkpoints/best.pth")
    trainer.load_checkpoint(checkpoint=state)
    # trainer.load_checkpoint("1_det_face_logs_2022-04-23 12_32_35/checkpoints/latest.pth")
    trainer.train(1, 1)


if __name__ == '__main__':
    gen_trainer(conf)
