from .center import *
from .triplet import *
from .arcface import *
from .attribute import *


import torch.nn.functional as F


def make_loss(config, num_classes):    # modified by gu

    sampler = config.sampler
    if config.loss_type == 'triplet':
        triplet = TripletLoss(config.margin)  # triplet loss
    elif config.loss_type == 'triplet_attr':
        triplet = TripletLoss(config.margin)
        attrweight = TripletLossAttrWeightes(margin=config.margin, dis_type="euclid")

    else:
        print('expected METRIC_LOSS_TYPE should be triplet'
              'but got {}'.format(config.loss_type))

    if config.label_smooth == 'on':
        xent = CrossEntropyLabelSmooth(
            num_classes=num_classes)     # new add by luo
        print("label smooth on, numclasses:", num_classes)

    if sampler == 'softmax':
        def loss_func(score, feat, target, index, attr=[]):
            return F.cross_entropy(score, target)
    elif config.sampler == 'triplet':
        def loss_func(score, feat, target, index, attr=[]):
            return triplet(feat[index], target[index])[0]
    elif config.sampler == 'softmax_triplet':
        def loss_func(score, feat, target, index=None, attr=[]):
            if config.loss_type == 'triplet':
                if config.label_smooth == 'on':
                    return xent(score, target) + triplet(feat, target)[0]
                else:
                    return F.cross_entropy(score, target) + triplet(feat, target)[0]
            elif config.loss_type == 'triplet_attr':
                if config.label_smooth == 'on':
                    return xent(score, target) + (triplet(feat, target)[0] + attrweight(feat, target, attr)) / 2
                else:
                    return F.cross_entropy(score, target) + (triplet(feat, target)[0] + attrweight(feat, target, attr)) / 2
            else:
                print('expected METRIC_LOSS_TYPE should be triplet or triplet_attr, '
                      'but got {}'.format(config.loss_type))
    else:
        print('expected sampler should be softmax, triplet or softmax_triplet, '
              'but got {}'.format(config.sampler))
    return loss_func


def make_loss_with_center(config, num_classes):    # modified by gu
    if config.model_name == 'resnet18' or config.model_name == 'resnet34':
        feat_dim = 512

    elif config.model_name == 'efficientnet_v2':
        feat_dim = 1280

    else:
        feat_dim = 2048

    if config.loss_type == 'center':
        center_criterion = CenterLoss(
            num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss

    elif config.loss_type == 'triplet_center':
        triplet = TripletLoss(config.margin)  # triplet loss
        center_criterion = CenterLoss(
            num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss
    elif config.loss_type == 'triplet_center_attr':
        triplet = TripletLoss(config.margin)  # triplet loss
        center_criterion = CenterLoss(
            num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss
        attrweight = TripletLossAttrWeightes(margin=config.margin, dis_type="euclid")   # attribute loss
    else:
        print('expected METRIC_LOSS_TYPE with center should be center, triplet_center, triplet_center_attr, '
              'but got {}'.format(config.loss_type))

    if config.label_smooth == 'on':
        xent = CrossEntropyLabelSmooth(
            num_classes=num_classes)     # new add by luo
        print("label smooth on, numclasses:", num_classes)

    def loss_func(score, feat, target, index=None, attr=[]):
        if config.loss_type == 'center':
            if config.label_smooth == 'on':
                return xent(score, target) + \
                    config.center_loss_weight * \
                    center_criterion(feat, target)
            else:
                return F.cross_entropy(score, target) + \
                    config.center_loss_weight * \
                    center_criterion(feat, target)

        elif config.loss_type == 'triplet_center':
            if config.label_smooth == 'on':
                return xent(score, target) + \
                    triplet(feat, target)[0] + \
                    config.center_loss_weight * \
                    center_criterion(feat, target)
            else:
                return F.cross_entropy(score, target) + \
                    triplet(feat, target)[0] + \
                    config.center_loss_weight * \
                    center_criterion(feat, target)
        
        elif config.loss_type == 'triplet_center_attr':
            if config.label_smooth == 'on':
                return xent(score, target) + \
                    (triplet(feat, target)[0] + \
                    config.center_loss_weight * \
                    center_criterion(feat, target) + \
                    attrweight(feat, target, attr)) / 3
            else:
                return F.cross_entropy(score, target) + \
                    (triplet(feat, target)[0] + \
                    config.center_loss_weight * \
                    center_criterion(feat, target) + \
                    attrweight(feat, target, attr)) / 3

        else:
            print('expected METRIC_LOSS_TYPE with center should be center, triplet_center, triplet_center_attr, '
                  'but got {}'.format(config.loss_type))
    return loss_func, center_criterion
