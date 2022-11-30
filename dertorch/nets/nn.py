import torch
from torch import nn

from .backbones.resnet import ResNet, BasicBlock, Bottleneck
from .backbones.resnet_ibn import *
from .backbones.efficientnet_v2 import *
import torch.nn.functional as F


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

class EfficientNet(torch.nn.Module):
    """
    EfficientNetV2: Smaller Models and Faster Training
    https://arxiv.org/pdf/2104.00298.pdf
    """

    def __init__(self, args, num_class=1000) -> None:
        super().__init__()
        gate_fn = [True, False]
        filters = [24, 48, 64, 128, 160, 272, 1792]

        feature = [Conv(args, 3, filters[0], torch.nn.SiLU(), 3, 2)]
        if args:
            filters[5] = 256
            filters[6] = 1280
        for i in range(2):
            if i == 0:
                feature.append(
                    Residual(args, filters[0], filters[0], 1, 1, gate_fn[0]))
            else:
                feature.append(
                    Residual(args, filters[0], filters[0], 1, 1, gate_fn[0]))

        for i in range(4):
            if i == 0:
                feature.append(
                    Residual(args, filters[0], filters[1], 2, 4, gate_fn[0]))
            else:
                feature.append(
                    Residual(args, filters[1], filters[1], 1, 4, gate_fn[0]))

        for i in range(4):
            if i == 0:
                feature.append(
                    Residual(args, filters[1], filters[2], 2, 4, gate_fn[0]))
            else:
                feature.append(
                    Residual(args, filters[2], filters[2], 1, 4, gate_fn[0]))

        for i in range(6):
            if i == 0:
                feature.append(
                    Residual(args, filters[2], filters[3], 2, 4, gate_fn[1]))
            else:
                feature.append(
                    Residual(args, filters[3], filters[3], 1, 4, gate_fn[1]))

        for i in range(9):
            if i == 0:
                feature.append(
                    Residual(args, filters[3], filters[4], 1, 6, gate_fn[1]))
            else:
                feature.append(
                    Residual(args, filters[4], filters[4], 1, 6, gate_fn[1]))

        for i in range(15):
            if i == 0:
                feature.append(
                    Residual(args, filters[4], filters[5], 2, 6, gate_fn[1]))
            else:
                feature.append(
                    Residual(args, filters[5], filters[5], 1, 6, gate_fn[1]))
        feature.append(Conv(args, filters[5], filters[6], torch.nn.SiLU()))

        self.feature = torch.nn.Sequential(*feature)

        initialize_weights(self)

    def forward(self, x):
        x = self.feature(x)
        return x
    
    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)['model'].state_dict()

        for i in param_dict:
            if 'fc' in i:
                continue
            self.state_dict()[i].copy_(param_dict[i])

class MultiAttributeRecogModuleBCE(nn.Module):
    def __init__(self, in_planes, num_classes=[]):
        super(MultiAttributeRecogModuleBCE, self).__init__()
        self.in_planes = in_planes
        self.out_planes = 16*8
        self.conv = nn.Conv2d(self.in_planes, self.in_planes, 1)
        weights_init_kaiming(self.conv)
        self.bn = nn.BatchNorm2d(self.in_planes)
        weights_init_kaiming(self.bn)
        self.attention_s_conv = nn.Conv2d(self.in_planes, 1, 3, padding=1)
        weights_init_kaiming(self.attention_s_conv)
        self.attention_t_conv = nn.Conv1d(self.out_planes, self.out_planes, 3, padding=1)
        weights_init_kaiming(self.attention_t_conv)

        self.classifier = nn.Linear(in_planes, sum(num_classes))
        self.classifier.apply(weights_init_classifier)
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x, b, t):
        local_feature = self.conv(x)
        local_feature = F.relu(self.bn(local_feature))
        a = self.attention_s_conv(local_feature)
        a = a.view(b, t, -1)
        a = a.permute(0, 2, 1)
        # a = self.attention_t_conv(a)
        # a = a.permute(0, 2, 1)
        a = torch.sigmoid(a)
        a = a.reshape(b, 1, 1, x.size(2), x.size(3))

        atten = a.expand(b, t, self.in_planes, x.size(2), x.size(3))
        global_feat = atten * x.view(b, 1, self.in_planes, x.size(2), x.size(3))
        global_feat = global_feat.view(b * t, self.in_planes, global_feat.size(3), global_feat.size(4))
        global_feat = self.gap(global_feat)
        global_feat = global_feat.view(b, t, -1)
        global_feat = F.relu(torch.mean(global_feat, 1))
        if self.training:
            y = self.classifier(global_feat)
        else:
            y = None
        return y, atten


class Backbone(nn.Module):
    """
    Resnet and EfficientNet-v2 backbones
    """

    def __init__(self, num_classes, model_name, model_path="", last_stride=1, neck="bnneck", neck_feat="after", pretrain_choice="", attr_lens = [], training=True, args=True):
        super(Backbone, self).__init__()

        if model_name == 'resnet18':
            self.in_planes = 512
            self.base = ResNet(last_stride=last_stride,
                               block=BasicBlock,
                               layers=[2, 2, 2, 2])
        elif model_name == 'resnet34':
            self.in_planes = 512
            self.base = ResNet(last_stride=last_stride,
                               block=BasicBlock,
                               layers=[3, 4, 6, 3])
        elif model_name == 'resnet50':
            self.in_planes = 2048
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3])
        elif model_name == 'resnet101':
            self.in_planes = 2048
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 23, 3])
        elif model_name == 'resnet152':
            self.in_planes = 2048
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 8, 36, 3])

        elif model_name == 'resnet50_ibn_a':
            self.in_planes = 2048
            self.base = resnet50_ibn_a(last_stride)

        elif model_name == 'efficientnet_v2':
            self.in_planes = 1280
            self.base = EfficientNet(args)

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......')

        self.gap = nn.AdaptiveAvgPool2d(1)
        # self.gap = nn.AdaptiveMaxPool2d(1)
        self.num_classes = num_classes
        self.neck = neck
        self.neck_feat = neck_feat
        self.training = training
        self.model_name = model_name
        self.attr_lens = attr_lens

        if len(self.attr_lens) != 0:
            self.attention_conv = nn.Conv2d(self.in_planes, 1, 3, padding=1)
            weights_init_kaiming(self.attention_conv)
            self.multi_attr_recoger = MultiAttributeRecogModuleBCE(self.in_planes, self.attr_lens)
            self.attr_on = True
        else: 
            self.attr_on = False

        if self.neck == 'no':
            self.classifier = nn.Linear(self.in_planes, self.num_classes)
            # self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            # self.classifier.apply(weights_init_classifier)
        elif self.neck == 'bnneck':
            self.bottleneck = nn.BatchNorm1d(self.in_planes)
            self.bottleneck.bias.requires_grad_(False)  # no shift
            self.classifier = nn.Linear(
                self.in_planes, self.num_classes, bias=False)

            self.bottleneck.apply(weights_init_kaiming)
            self.classifier.apply(weights_init_classifier)

    def forward(self, x):
        attr_ys = []
        b = x.size(0) 
        t = 8 
        # x = x.view(b * t, 1, x.size(2), x.size(3))
        global_feat = self.base(x)

        if self.attr_on:
            attr_y, attr_a = self.multi_attr_recoger(global_feat.detach(), b, t)
            attr_ys.append(attr_y)

            a = self.attention_conv(global_feat)
            a = a.view(b, 1, 1, global_feat.size(2), global_feat.size(3))
            a = torch.sigmoid(a)
            # a = torch.nn.functional.softmax(a, 1)
            a = a.expand(b, t, self.in_planes, global_feat.size(2), global_feat.size(3))
            global_feat = a * global_feat.view(b, 1, self.in_planes, global_feat.size(2), global_feat.size(3))
            global_feat = global_feat.view(b * t, self.in_planes, global_feat.size(3), global_feat.size(4))

            global_feat1 = attr_a * global_feat.view(b, t, self.in_planes, global_feat.size(2), global_feat.size(3))
            global_feat1 = global_feat1.view(b * t, self.in_planes, global_feat.size(2), global_feat.size(3))
            global_feat = torch.cat([global_feat, global_feat1], 1)

        global_feat = self.gap(self.base(x))  # (b, in_planes, 1, 1)
        global_feat = global_feat.view(
            global_feat.shape[0], -1)  # flatten to (bs, in_planes)

        if self.neck == 'no':
            feat = global_feat
        elif self.neck == 'bnneck':
            # normalize for angular softmax
            feat = self.bottleneck(global_feat)

        if self.training:
            cls_score = self.classifier(feat)
            return cls_score, global_feat, attr_ys  # global feature for triplet loss
        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return feat, attr_ys
            else:
                # print("Test with feature before BN")
                return global_feat, attr_ys

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path).state_dict()
        for i in param_dict:
            if 'classifier' in i:
                continue
            if 'attention_conv' in i:
                continue
            if 'multi_attr_recoger' in i:
                continue
            if 'module' in i:
                self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
                continue
            self.state_dict()[i].copy_(param_dict[i])

    