"""
Handles the training
"""
import math
import sys
from time import perf_counter
from datetime import datetime

import torch

from globals import DEVICE
from model import get_resnet50_model, get_sgd_optimizer, get_dataloader
from logger import EpochLogger

def _warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
    """
    Warmup learning rate scheduler
    """
    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)

def _train_epoch(model, optimizer, dataloader, epoch):
    """
    Trains the epoch
    """
    model.train()

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(dataloader) - 1)

        lr_scheduler = _warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    logger = EpochLogger('Training', epoch, 0, len(dataloader))

    avg_loss = []
    time_delta = []

    i = 0

    for images, targets in dataloader:
        time_start = perf_counter()

        images = list(i.to(DEVICE) for i in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets) # ~1s op

        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()

        if not math.isfinite(loss_value):
            print(f'Loss is {loss_value}, stopping training')
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward() # ~1.5s op
        optimizer.step()

        avg_loss.append(loss_value)
        time_delta.append(perf_counter() - time_start)

        logger.update(loss=f'{(sum(avg_loss) / 10):.4f}')
        logger.update(lr=f'{(optimizer.param_groups[0]["lr"]):.6f}')
        logger.update(time=f'{(sum(time_delta)):.4f}')
        logger.log()

        avg_loss.clear()
        time_delta.clear()

        i += 1

        if i % 10 == 0:
            lr_scheduler.step()

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

def _evaluate_epoch(model, dataloader, epoch):
    """
    Evaluates the epoch
    """
    model.eval()
    logger = EpochLogger('Testing', epoch, 0, len(dataloader))

    with torch.no_grad():
        for images, targets in dataloader:
            time_start = perf_counter()

            images = list(image.to(DEVICE) for image in images)
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            predictions = model(images)

            logger.update(prediction=predictions[0]['labels'])
            logger.update(target=targets[0]['labels'])
            logger.update(time=perf_counter() - time_start)
            logger.log()

def train():
    """
    Run training
    """
    print('[Training]')
    print(f'Using device: {DEVICE}')

    model = get_resnet50_model()
    optimizer = get_sgd_optimizer(model)
    dataloader, dataloader_test = get_dataloader(
        train_batch_size=1, test_batch_size=1, take_one=False
    )

    epochs = 1
    print(f'Number of epochs: {epochs}')

    for epoch in range(epochs):
        _train_epoch(model, optimizer, dataloader, epoch)
        _evaluate_epoch(model, dataloader_test, epoch)

    torch.save(model.state_dict(), f'./models/model_{datetime.now()}.pth')
