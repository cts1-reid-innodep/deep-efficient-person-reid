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
        img, target, _, attrs = batch
        img = img.to(device) if torch.cuda.device_count() >= 1 else img
        target = target.to(
            device) if torch.cuda.device_count() >= 1 else target
        score, feat, attr = model(img)

        with torch.cuda.amp.autocast(enabled=config.mixed_precision):
            amp_scale = torch.cuda.amp.GradScaler()
            loss = loss_fn(score, feat, target)
            # print("Total loss is {}".format(
            #     loss))

        if len(config.attr_lens) != 0:
            attrs = [a[0].cuda() for a in attrs]
            attr_criter = BCEWithLogitsLoss()
            attr_loss = attr_criter(attr[0][0], attrs[0])
            for i in range(1, len(attrs)):
                attr_loss += attr_criter(attr[0][i], attrs[i])
            loss += attr_loss

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
        img, target, _, attrs = batch
        img = img.to(device) if torch.cuda.device_count() >= 1 else img
        target = target.to(
            device) if torch.cuda.device_count() >= 1 else target
        score, feat, attr = model(img)

        with torch.cuda.amp.autocast(enabled=config.mixed_precision):
            amp_scale = torch.cuda.amp.GradScaler()
            loss = loss_fn(score, feat, target, attr=attr)
            # print("Total loss is {}, center loss is {}".format(
            #     loss, center_criterion(feat, target)))

        if len(config.attr_lens) != 0:
            attrs = [a[0].cuda() for a in attrs]
            attr_criter = BCEWithLogitsLoss()
            attr_loss = attr_criter(attr[0][0], attrs[0])
            for i in range(1, len(attrs)):
                attr_loss += attr_criter(attr[0][i], attrs[i])
            loss += attr_loss

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
            feat, _ = model(data)

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
    trainer = create_supervised_trainer(
        config, model, optimizer, loss_fn, device=device)
    evaluator = create_supervised_evaluator(model, metrics={'r1_mAP': R1_mAP(
        num_query, max_rank=50, feat_norm=config.test_feat_norm)}, device=device)
    checkpointer = ModelCheckpoint(
        output_dir, config.model_name, checkpoint_period, n_saved=10, require_empty=False)
    timer = Timer(average=True)

    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {'model': model,
                                                                     'optimizer': optimizer})
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
            print('Trainer eval')
            evaluator.run(val_loader)
            cmc, mAP, _, _, _, _, _ = evaluator.state.metrics['r1_mAP']
            logger.info(
                "Validation Results - Epoch: {}".format(engine.state.epoch))
            logger.info("mAP: {:.1%}".format(mAP))
            for r in [1, 5, 10]:
                logger.info(
                    "CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))

    trainer.run(train_loader, max_epochs=epochs)

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