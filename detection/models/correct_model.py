from torchvision import models
from torch import nn
import torch
import torch.nn.functional as F
from .losses import quadrilateral_grad_loss

HIDDEN_DIMS = 8
OUTPUT_DIMS = 9 + 4  # 透视变换9维+代码区4坐标4维


class CorrectModel(nn.Module):
    def __init__(self, model_type, pretrained=True):
        super().__init__()
        if model_type == "wide_resnet50_2":
            model = models.wide_resnet50_2(pretrained=pretrained)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, HIDDEN_DIMS)
        elif model_type == "wide_resnet101_2":
            model = models.wide_resnet101_2(pretrained=pretrained)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, HIDDEN_DIMS)
            model._fc = nn.Linear(1280, HIDDEN_DIMS)
        elif model_type == 'inception_v3_google':
            model = models.inception_v3(pretrained=True, aux_logits=False)
            model.fc = nn.Linear(model.fc.in_features, HIDDEN_DIMS)
        elif model_type == 'resnet50':
            model = models.resnet50(pretrained=True, )
            model.fc = nn.Linear(model.fc.in_features, HIDDEN_DIMS)
        elif model_type == 'resnet152':
            model = models.resnet152(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, HIDDEN_DIMS)
        elif model_type == 'resnext101_32x8d':
            model = models.resnext101_32x8d(pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, HIDDEN_DIMS)
        elif model_type == 'densenet121':
            model = models.densenet121(pretrained=True)
            model.classifier = nn.Linear(model.classifier.in_features,
                                         HIDDEN_DIMS)
        elif model_type == 'densenet201':
            model = models.densenet201(pretrained=True)
            model.classifier = nn.Linear(model.classifier.in_features,
                                         HIDDEN_DIMS)
        elif model_type == 'mobilenet_v2':
            model = models.mobilenet_v2(pretrained=True)
            model.classifier._modules['1'] = nn.Linear(model.classifier._modules['1'].in_features,
                                                       HIDDEN_DIMS)
        elif model_type == 'mnasnet0_5':
            model = models.mnasnet0_5(pretrained=True)
            model.classifier._modules['1'] = nn.Linear(model.classifier._modules['1'].in_features,
                                                       HIDDEN_DIMS)
        elif model_type == 'mnasnet0_75':
            model = models.mnasnet0_5(pretrained=True)
            model.classifier._modules['1'] = nn.Linear(model.classifier._modules['1'].in_features,
                                                       HIDDEN_DIMS)
        elif model_type == 'squeezenet1_0':
            model = models.squeezenet1_0(pretrained=True)
            model.classifier.final_conv = nn.Conv2d(1000, HIDDEN_DIMS, kernel_size=1)
        elif model_type == 'vgg16':
            model = models.vgg16(pretrained=True)
            model.classifier._modules['6'] = nn.Linear(model.classifier._modules['6'].in_features,
                                                       HIDDEN_DIMS)
        elif model_type == 'resnext50_32x4d':
            model = models.resnext50_32x4d(pretrained=True)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, HIDDEN_DIMS)
        # elif model_type == 'senet':
        #     model = se_resnext101_32x4d(OUTPUT_DIMS=OUTPUT_DIMS)
        else:
            raise ValueError("Don't suppert this model_type!")
        self.backbone = model
        # self.perspective_head = nn.Linear(HIDDEN_DIMS, 9)
        # self.bbox_head = nn.Linear(HIDDEN_DIMS, 4)

    def forward(self, x, target=None, mask=None):
        pred = self.backbone(x)
        pred = pred.reshape(-1, 4, 2)
        # perspective_pred = self.perspective_head(pred)
        # bbox_pred = self.bbox_head(pred)
        if target is not None:
            # perspective_target = target[:, :9]
            # bbox_target = target[:, 9:]
            # per_loss = F.smooth_l1_loss(perspective_pred, perspective_target)
            # if torch.any(~mask):
            #     print()
            bbox_loss = F.l1_loss(pred, target, reduction="none").sum(-1)
            bbox_loss = (bbox_loss[mask]).mean()

            grad_loss = quadrilateral_grad_loss(pred, target, mask)
            # grad_loss = grad_loss * 0.1
            loss = grad_loss + bbox_loss
            ret_v = {
                "loss": loss,
                "grad_loss": grad_loss,
                "box_loss": bbox_loss
            }
            return ret_v
        else:
            return pred