import logging

import torch
import torch.nn as nn
from utils.getter import *
from bisect import bisect_right
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, Timer
from ignite.metrics import RunningAverage
from PIL import Image
from tqdm import tqdm
from augmentations.transform import build_transforms
from torch.optim import SGD, AdamW
import os
import numpy as np

from metrics.mAP import R1_mAP

from torch.nn.modules import Module
from torch.nn import functional as F
from torch.nn import _reduction as _Reduction

from torch import Tensor
from typing import Callable, Optional

global ITER
ITER = 0

class _Loss(Module):
    reduction: str

    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean') -> None:
        super(_Loss, self).__init__()
        if size_average is not None or reduce is not None:
            self.reduction: str = _Reduction.legacy_get_string(size_average, reduce)
        else:
            self.reduction = reduction

class BCEWithLogitsLoss(_Loss):
    r"""This loss combines a `Sigmoid` layer and the `BCELoss` in one single
    class. This version is more numerically stable than using a plain `Sigmoid`
    followed by a `BCELoss` as, by combining the operations into one layer,
    we take advantage of the log-sum-exp trick for numerical stability.

    The unreduced (i.e. with :attr:`reduction` set to ``'none'``) loss can be described as:

    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = - w_n \left[ y_n \cdot \log \sigma(x_n)
        + (1 - y_n) \cdot \log (1 - \sigma(x_n)) \right],

    where :math:`N` is the batch size. If :attr:`reduction` is not ``'none'``
    (default ``'mean'``), then

    .. math::
        \ell(x, y) = \begin{cases}
            \operatorname{mean}(L), & \text{if reduction} = \text{`mean';}\\
            \operatorname{sum}(L),  & \text{if reduction} = \text{`sum'.}
        \end{cases}

    This is used for measuring the error of a reconstruction in for example
    an auto-encoder. Note that the targets `t[i]` should be numbers
    between 0 and 1.

    It's possible to trade off recall and precision by adding weights to positive examples.
    In the case of multi-label classification the loss can be described as:

    .. math::
        \ell_c(x, y) = L_c = \{l_{1,c},\dots,l_{N,c}\}^\top, \quad
        l_{n,c} = - w_{n,c} \left[ p_c y_{n,c} \cdot \log \sigma(x_{n,c})
        + (1 - y_{n,c}) \cdot \log (1 - \sigma(x_{n,c})) \right],

    where :math:`c` is the class number (:math:`c > 1` for multi-label binary classification,
    :math:`c = 1` for single-label binary classification),
    :math:`n` is the number of the sample in the batch and
    :math:`p_c` is the weight of the positive answer for the class :math:`c`.

    :math:`p_c > 1` increases the recall, :math:`p_c < 1` increases the precision.

    For example, if a dataset contains 100 positive and 300 negative examples of a single class,
    then `pos_weight` for the class should be equal to :math:`\frac{300}{100}=3`.
    The loss would act as if the dataset contains :math:`3\times 100=300` positive examples.

    Examples::

        >>> target = torch.ones([10, 64], dtype=torch.float32)  # 64 classes, batch size = 10
        >>> output = torch.full([10, 64], 1.5)  # A prediction (logit)
        >>> pos_weight = torch.ones([64])  # All weights are equal to 1
        >>> criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        >>> criterion(output, target)  # -log(sigmoid(1.5))
        tensor(0.2014)

    Args:
        weight (Tensor, optional): a manual rescaling weight given to the loss
            of each batch element. If given, has to be a Tensor of size `nbatch`.
        size_average (bool, optional): Deprecated (see :attr:`reduction`). By default,
            the losses are averaged over each loss element in the batch. Note that for
            some losses, there are multiple elements per sample. If the field :attr:`size_average`
            is set to ``False``, the losses are instead summed for each minibatch. Ignored
            when :attr:`reduce` is ``False``. Default: ``True``
        reduce (bool, optional): Deprecated (see :attr:`reduction`). By default, the
            losses are averaged or summed over observations for each minibatch depending
            on :attr:`size_average`. When :attr:`reduce` is ``False``, returns a loss per
            batch element instead and ignores :attr:`size_average`. Default: ``True``
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the sum of the output will be divided by the number of
            elements in the output, ``'sum'``: the output will be summed. Note: :attr:`size_average`
            and :attr:`reduce` are in the process of being deprecated, and in the meantime,
            specifying either of those two args will override :attr:`reduction`. Default: ``'mean'``
        pos_weight (Tensor, optional): a weight of positive examples.
                Must be a vector with length equal to the number of classes.

    Shape:
        - Input: :math:`(*)`, where :math:`*` means any number of dimensions.
        - Target: :math:`(*)`, same shape as the input.
        - Output: scalar. If :attr:`reduction` is ``'none'``, then :math:`(*)`, same
          shape as input.

     Examples::

        >>> loss = nn.BCEWithLogitsLoss()
        >>> input = torch.randn(3, requires_grad=True)
        >>> target = torch.empty(3).random_(2)
        >>> output = loss(input, target)
        >>> output.backward()
    """
    def __init__(self, weight: Optional[Tensor] = None, size_average=None, reduce=None, reduction: str = 'mean',
                 pos_weight: Optional[Tensor] = None) -> None:
        super(BCEWithLogitsLoss, self).__init__(size_average, reduce, reduction)
        self.register_buffer('weight', weight)
        self.register_buffer('pos_weight', pos_weight)
        self.weight: Optional[Tensor]
        self.pos_weight: Optional[Tensor]

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.binary_cross_entropy_with_logits(input, target,
                                                  self.weight,
                                                  pos_weight=self.pos_weight,
                                                  reduction=self.reduction)

def create_supervised_trainer(config, model, optimizer, loss_fn,
                              device=None):
    """
    Factory function for creating a trainer for supervised models
    Args:
        model (`torch.nn.Module`): the model to train
        optimizer (`torch.optim.Optimizer`): the optimizer to use
        loss_fn (torch.nn loss function): the loss function to use
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.
    Returns:
        Engine: a trainer engine with supervised update function
    """
    if device:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)

    def _update(engine, batch):
        model.train()
        img, target, img_path = batch
        img = img.to(device) if torch.cuda.device_count() >= 1 else img
        target = target.to(
            device) if torch.cuda.device_count() >= 1 else target
        score, feat = model(img)

        with torch.cuda.amp.autocast(enabled=config.mixed_precision):
            amp_scale = torch.cuda.amp.GradScaler()
            loss = loss_fn(score, feat, target)
            # print("Total loss is {}".format(
            #     loss))

        if config.mixed_precision:
            optimizer.zero_grad()
            amp_scale.scale(loss).backward()
            amp_scale.step(optimizer)
            amp_scale.update()

        else:
            loss.backward()
            optimizer.step()

        # compute acc
        acc = (score.max(1)[1] == target).float().mean()
        return loss.item(), acc.item()

    return Engine(_update)


def create_supervised_trainer_with_center(config, model, center_criterion, optimizer, optimizer_center, loss_fn, center_loss_weight,
                                          device=None):
    if device:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)

    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        optimizer_center.zero_grad()
        img, target = batch
        img = img.to(device) if torch.cuda.device_count() >= 1 else img
        target = target.to(
            device) if torch.cuda.device_count() >= 1 else target
        score, feat = model(img)

        with torch.cuda.amp.autocast(enabled=config.mixed_precision):
            amp_scale = torch.cuda.amp.GradScaler()
            loss = loss_fn(score, feat, target)
            # print("Total loss is {}, center loss is {}".format(
            #     loss, center_criterion(feat, target)))

        if config.mixed_precision:
            optimizer.zero_grad()
            amp_scale.scale(loss).backward()
            amp_scale.step(optimizer)

        else:
            loss.backward()
            optimizer.step()

        for param in center_criterion.parameters():
            param.grad.data *= (1. / center_loss_weight)

        if config.mixed_precision:
            amp_scale.step(optimizer_center)
            amp_scale.update()
        else:
            optimizer_center.step()

        # compute acc
        acc = (score.max(1)[1] == target).float().mean()
        return loss.item(), acc.item()

    return Engine(_update)


def create_supervised_evaluator(model, metrics,
                                device=None):
    """
    Factory function for creating an evaluator for supervised models
    Args:
        model (`torch.nn.Module`): the model to train
        metrics (dict of str - :class:`ignite.metrics.Metric`): a map of metric names to Metrics
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.
    Returns:
        Engine: an evaluator engine with supervised inference function
    """
    if device:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)

    def inference(engine, batch):
        model.eval()
        with torch.no_grad():
            data, pids, camids, _ = batch
            data = data.to(device) if torch.cuda.device_count() >= 1 else data
            feat = model(data)

            return feat, pids, camids

    engine = Engine(inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


def do_train(
        config,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        loss_fn,
        num_query,
        start_epoch,
        dataset_reference,
        train_loader_reference,
        num_classes
):
    log_period = config.log_period
    checkpoint_period = config.checkpoint_period
    eval_period = config.eval_period
    output_dir = config.output_dir
    device = config.device
    epochs = config.num_epochs
    timer = Timer(average=True)
    logger = logging.getLogger("reid_baseline.train")
    logger.info("Start training")

    top = int(start_epoch/40) # the choose of the nearest sample
    top_update = start_epoch # the first iteration train 80 steps and the following train 40

    # Train and test
    for epoch in range(start_epoch, epochs):
        # get nearest samples and reset the model
        if top_update < 80:
            train_step = 80
        else:
            train_step = 40
        if top_update % train_step == 0:
            print("top: ", top)
            A, path_labeled = PSP(model, train_loader_reference, train_loader, top, config)
            top += 1
            model = Backbone(num_classes=num_classes, model_name=config.model_name, model_path=config.pretrain_path, pretrain_choice=config.pretrain_choice, attr_lens=config.attr_lens).to(config.device)
            optimizer = get_lr_policy(config.lr_policy, model)
            scheduler = WarmupMultiStepLR(optimizer, config.steps, config.gamma, config.warmup_factor,
                                      config.warmup_iters, config.warmup_method)
            A_store = A.clone()
        top_update += 1

        for data in tqdm(train_loader, desc='Iteration', leave=False):
            model.train()
            global ITER
            ITER += 1
            images, labels_batch, img_path, attrs = data
            index, index_labeled = find_index_by_path(img_path, dataset_reference.train, path_labeled)
            images_relevant, GCN_index, choose_from_nodes, labels = load_relevant(config, dataset_reference.train, index, A_store, labels_batch, index_labeled)
            # if device:
            model.to(device)
            images = images_relevant.to(device)

            scores, feat, attr = model(images)
            del images
            loss = loss_fn(scores, feat, labels.to(device), choose_from_nodes, attr)

            if len(config.attr_lens) != 0:
                attrs = [a[0].cuda() for a in attrs]
                attr_criter = BCEWithLogitsLoss()
                attr_loss = attr_criter(attr[0][0], attrs[0])
                for i in range(1, len(attrs)):
                    attr_loss += attr_criter(attr[0][i], attrs[i])
                loss += attr_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = (scores[choose_from_nodes].max(1)[1].cpu() == labels_batch).float().mean()
            if ITER % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                        .format(epoch+1, ITER, len(train_loader),
                                loss.item(), acc.item(),
                                scheduler.get_lr()[0]))
            if len(train_loader) == ITER:
                ITER = 0

        logger.info('Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]'
                    .format(epoch+1, timer.value() * timer.step_count,
                            train_loader.batch_size / timer.value()))
        logger.info('-' * 10)
        timer.reset()

        if (epoch+1) % checkpoint_period == 0:
            model_name = f'resnet50_ibn_a_model_{epoch+1}'
            optimizer_name = f'resnet50_ibn_a_optimizer_{epoch+1}'
            torch.save(model, os.path.join(output_dir, model_name)+".pth")
            torch.save(optimizer, os.path.join(output_dir, optimizer_name)+".pth")

        # Validation
        if (epoch+1) % eval_period == 0:
            print('Trainer eval')
            all_feats = []
            all_pids = []
            all_camids = []
            for data in tqdm(val_loader, desc='Feature Extraction', leave=False):
                model.eval()
                with torch.no_grad():
                    images, pids, camids, _ = data

                    model.to(device)
                    images = images.to(device)

                    feats, _ = model(images)
                    del images
                all_feats.append(feats.cpu())
                all_pids.extend(np.asarray(pids))
                all_camids.extend(np.asarray(camids))

            cmc, mAP = evaluation(all_feats, all_pids, all_camids, num_query, rr=config.test_reranking)
            logger.info(
                "Validation Results - Epoch: {}".format(epoch+1))
            logger.info("mAP: {:.1%}".format(mAP))
            for r in [1, 5, 10]:
                logger.info(
                    "CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))

        scheduler.step()

def PSP(model, train_loader, train_loader_orig, top, cfg):
    vis = len(train_loader_orig.dataset)
    A_base = torch.zeros(vis, len(train_loader.dataset)) # the one-shot example
    A_map = torch.zeros(vis, len(train_loader.dataset))

    if top == 0: # no PSP choose
        img_paths = []
        for data in tqdm(train_loader):
            images, label, img_path, _ = data
            img_paths += img_path
    else:
        device = cfg.device
        model.eval().to(device)
        feats = []
        labels = []
        # 1 get all features and distance
        img_paths = []
        with torch.no_grad():
            for data in tqdm(train_loader):
                images, label, img_path, _ = data
                images = images.to(device)
                feat, _ = model(images)
                feats.append(feat.cpu())
                labels.append(label)
                img_paths += img_path
        labels = torch.cat(labels, dim=0)
        feats = torch.cat(feats, dim=0)

    pathes_labeded = []
    all_labels = []
    # only use for accuracy estimate
    for unlabed_data in train_loader_orig:
        images, label, img_path, _ = unlabed_data
        pathes_labeded += img_path
        all_labels.append(label)

    index = {}
    index_list = []
    for unlabeled_one_shot_index, img_path in enumerate(pathes_labeded):
        for index_origin, path_of_origin in enumerate(img_paths):
            if img_path.split("/")[-1] == path_of_origin.split("/")[-1]:
                index[index_origin] = unlabeled_one_shot_index
                index_list.append(index_origin)
                A_base[unlabeled_one_shot_index][index_origin] = 1
                break
    if top == 0:
        return A_base, pathes_labeded
    else:
        A_gt = torch.zeros(vis, len(labels))
        for count, label_each in enumerate(labels[index_list]):
            A_gt[count, labels == label_each] = 1

        # calculate distance
        dis_feats = get_euclidean_dist(feats, feats[index_list])

        dis_feats = -dis_feats + dis_feats.max()
        A = dis_feats

        no_eye_A = A - A_base * A

        test_top = top
        sorted_A = no_eye_A.to(device).sort(descending=True)[1][:, 0:test_top]
        for index_labeled, one_labeled in enumerate(sorted_A):
            for chosen_index, choose_one in enumerate(one_labeled):
                exist_index_top_e = False
                choose_from_top = no_eye_A[:, choose_one].sort(descending=True)[1][:1]
                for i in choose_from_top:
                    if i == index_labeled:
                        exist_index_top_e = True
                        break
                if (choose_one not in index.keys()) & exist_index_top_e:
                    A_map[index_labeled][choose_one] = 1
                    # A_map[choose_one][index_labeled] = 1

        # for test
        # acc = (A_gt - A_base)[A_map > 0]
        # print(acc.sum() / (A_map > 0).sum(),' ', (A_map > 0).sum())
        A_map = A_map + A_base
        return A_map, pathes_labeded

def get_euclidean_dist(gf, qf):
    m, n = qf.shape[0], gf.shape[0]
    distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
              torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(1, -2, qf, gf.t())
    return distmat

def find_index_by_path(path, data_origin, path_labeled=None):
    index = []
    index_labeled = []
    for img_path in path:
        max_index = img_path.split("/")[-1]
        for index_origin, path_of_origin in enumerate(data_origin):
            id_from_path = path_of_origin[0].split("/")[-1]
            if max_index == id_from_path:
                index.append(index_origin)
                break
        if path_labeled is None: continue
        for index_labeded, path_temp in enumerate(path_labeled):
            if max_index == path_temp.split("/")[-1]:
                index_labeled.append(index_labeded)
                break
    return index, index_labeled

def load_relevant(cfg, data_train, index_batch_withid, A_map, label_labeled, index_labeled=None):
    train_transforms = build_transforms(cfg, is_train=True)
    indices = get_indice_graph(A_map, index_batch_withid, 96, index_labeled)
    indices_to_index = {}
    images = []
    for counter, indice in enumerate(indices):
        img_path = data_train[indice][0]
        img_orig = Image.open(img_path).convert('RGB')
        img = train_transforms(img_orig)
        images.append(img)
        indices_to_index[indice] = counter
    images = torch.stack(images)

    choose_from_nodes = []
    for id in index_batch_withid:
        choose_from_nodes.append(indices_to_index[id])

    if index_labeled is None: return images, indices, choose_from_nodes, None
    labels = []
    for indice in indices:
        for id, each_labeled in zip(index_labeled, label_labeled):
            if (A_map[id][indice] > 0):
                labels.append(each_labeled)
                break
    labels = torch.stack(labels)

    return images, indices, choose_from_nodes, labels

def get_indice_graph(adj, mask, size, index_labeled):
    indices = mask
    pre_indices = set()
    indices = set(indices)
    choosen = indices if index_labeled is None else set(index_labeled)

    # pre_indices = indices.copy()
    candidates = get_candidates(adj, choosen) - indices
    if len(candidates) > size - len(indices):
        candidates = set(np.random.choice(list(candidates), size-len(indices), False))
    indices.update(candidates)
    # print('indices size:-------------->', len(indices))
    return sorted(indices)

def get_candidates(adj, new_add):
    same = adj[sorted(new_add)].sum(dim=0).nonzero().squeeze().numpy()
    return set(tuple(same))

def get_lr_policy(opt_config, model):
    optimizer_params = []

    if opt_config["name"] == 'sgd':
        optimizer_name = SGD
    elif opt_config["name"] == 'adam':
        optimizer_name = AdamW

    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        weight_decay = opt_config['weight_decay']
        lr = opt_config['lr']
        if "bias" in key:
            lr = opt_config['lr'] * opt_config['bias_lr_factor']
            weight_decay = opt_config['weight_decay_bias']

        if opt_config["name"] == 'sgd':
            optimizer_params += [{"params": [value], "lr": lr,
                                  "weight_decay": weight_decay,
                                  "momentum": opt_config['momentum'],
                                  "nesterov": True}]

        elif opt_config["name"] == 'adam':
            optimizer_params += [{"params": [value], "lr": lr,
                                  "weight_decay": weight_decay,
                                  'betas': (opt_config['momentum'], 0.999)}]

    optimizer = optimizer_name(optimizer_params)

    return optimizer

class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        milestones,
        gamma=0.1,
        warmup_factor=1.0 / 3,
        warmup_iters=500,
        warmup_method="linear",
        last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = self.last_epoch / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]

def evaluation(all_feats, all_pids, all_camids, num_query, rr=False, max_rank=50):
    all_feats = torch.cat(all_feats, dim=0)
    # query
    qf = all_feats[:num_query]
    q_pids = np.asarray(all_pids[:num_query])
    q_camids = np.asarray(all_camids[:num_query])
    # gallery
    gf = all_feats[num_query:]
    g_pids = np.asarray(all_pids[num_query:])
    g_camids = np.asarray(all_camids[num_query:])
    
    if rr != 'yes':    
        m, n = qf.shape[0], gf.shape[0]
        distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
                  torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
        distmat.addmm_(1, -2, qf, gf.t())
        distmat = distmat.cpu().numpy()
    else:
        print('Calculating Distance')
        q_g_dist = np.dot(qf.data.cpu(), np.transpose(gf.data.cpu()))
        q_q_dist = np.dot(qf.data.cpu(), np.transpose(qf.data.cpu()))
        g_g_dist = np.dot(gf.data.cpu(), np.transpose(gf.data.cpu()))
        print('Re-ranking:')
        distmat= re_ranking(q_g_dist, q_q_dist, g_g_dist)
        
    """Evaluation with market1501 metric
    Key: for each query identity, its gallery images from the same camera view are discarded.
    """
    num_q, num_g = distmat.shape
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query
    for q_idx in tqdm(range(num_q), desc='Metric Computing', leave=False):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc_base = orig_cmc.cumsum()
        cmc = cmc_base

        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        # tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        # tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        tmp_cmc = tmp_cmc / (np.arange(tmp_cmc.size) + 1.0)
        tmp_cmc = tmp_cmc * orig_cmc

        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP

def do_train_with_center(
        config,
        model,
        center_criterion,
        train_loader,
        val_loader,
        optimizer,
        optimizer_center,
        scheduler,
        loss_fn,
        num_query,
        start_epoch
):
    log_period = config.log_period
    checkpoint_period = config.checkpoint_period
    eval_period = config.eval_period
    output_dir = config.output_dir
    device = config.device
    epochs = config.num_epochs

    logger = logging.getLogger("reid_baseline.train")
    logger.info("Start training")
    trainer = create_supervised_trainer_with_center(
        config, model, center_criterion, optimizer, optimizer_center, loss_fn, config.center_loss_weight, device=device)
    evaluator = create_supervised_evaluator(model, metrics={'r1_mAP': R1_mAP(
        num_query, max_rank=50, feat_norm=config.test_feat_norm)}, device=device)
    checkpointer = ModelCheckpoint(
        output_dir, config.model_name, checkpoint_period, n_saved=10, require_empty=False)
    timer = Timer(average=True)

    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {'model': model,
                                                                     'optimizer': optimizer,
                                                                     'center_param': center_criterion,
                                                                     'optimizer_center': optimizer_center})

    timer.attach(trainer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                 pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)

    # average metric to attach on trainer
    RunningAverage(output_transform=lambda x: x[0]).attach(trainer, 'avg_loss')
    RunningAverage(output_transform=lambda x: x[1]).attach(trainer, 'avg_acc')

    @trainer.on(Events.STARTED)
    def start_training(engine):
        engine.state.epoch = start_epoch

    @trainer.on(Events.EPOCH_STARTED)
    def adjust_learning_rate(engine):
        scheduler.step()

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        global ITER
        ITER += 1

        if ITER % log_period == 0:
            logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                        .format(engine.state.epoch, ITER, len(train_loader),
                                engine.state.metrics['avg_loss'], engine.state.metrics['avg_acc'],
                                scheduler.get_lr()[0]))
        if len(train_loader) == ITER:
            ITER = 0

    # adding handlers using `trainer.on` decorator API
    @trainer.on(Events.EPOCH_COMPLETED)
    def print_times(engine):
        logger.info('Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]'
                    .format(engine.state.epoch, timer.value() * timer.step_count,
                            train_loader.batch_size / timer.value()))
        logger.info('-' * 10)
        timer.reset()

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        if engine.state.epoch % eval_period == 0:
            evaluator.run(val_loader)
            cmc, mAP, _ , _, _ , _, _ = evaluator.state.metrics['r1_mAP']
            logger.info(
                "Validation Results - Epoch: {}".format(engine.state.epoch))
            logger.info("mAP: {:.1%}".format(mAP))
            for r in [1, 5, 10]:
                logger.info(
                    "CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))

    trainer.run(train_loader, max_epochs=epochs)
