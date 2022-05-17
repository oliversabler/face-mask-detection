import math
import sys
import time

from datetime import datetime

import wandb
import torch
from globals import DEVICE
from model import get_resnet50_model, get_sgd_optimizer, get_dataloader
from utils import Logger


def _warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def _train_epoch(model, optimizer, dataloader, epoch):
    model.train()

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(dataloader) - 1)

        lr_scheduler = _warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    logger = Logger("Training", epoch)

    avg_loss = []
    time_delta = []

    i = 0

    for images, targets in dataloader:
        # Todo: use perf_counter
        time_start = time.time()

        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()

        wandb.log({"loss": loss_value})

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        avg_loss.append(loss_value)
        time_delta.append(time.time() - time_start)

        if logger.iteration % 10 == 0:
            if logger.iteration != 0:
                logger.update(loss="{:.4f}".format(sum(avg_loss) / 10))
                logger.update(lr="{:.6f}".format(optimizer.param_groups[0]["lr"]))
                logger.update(time="{:.4f}s".format(sum(time_delta)))
                print(logger)

                avg_loss.clear()
                time_delta.clear()

        logger.increment()

        if i % 10 == 0:
            if lr_scheduler is not None:
                lr_scheduler.step()

        i += 1

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)


def _evaluate_epoch(model, dataloader, epoch):
    model.eval()
    logger = Logger("Testing", epoch)

    with torch.no_grad():
        for images, targets in dataloader:
            # Todo: use perf_counter
            time_start = time.time()

            images = list(image.to(DEVICE) for image in images)
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            predictions = model(images)

            if logger.iteration % 10 == 0:
                logger.update(prediction=predictions[0]["labels"])
                logger.update(target=targets[0]["labels"])
                logger.update(time=time.time() - time_start)
                print(logger)

            logger.increment()


def train():
    print("[Training]")
    print(f"Using device: {DEVICE}")

    model = get_resnet50_model()
    optimizer = get_sgd_optimizer(model)
    dataloader, dataloader_test = get_dataloader(
        train_batch_size=1, test_batch_size=1, take_one=False
    )

    epochs = 10
    print(f"Number of epochs: {epochs}")

    wandb.init()

    for epoch in range(epochs):
        _train_epoch(model, optimizer, dataloader, epoch)
        _evaluate_epoch(model, dataloader_test, epoch)

    wandb.finish()

    torch.save(model.state_dict(), f"./models/model_{datetime.now()}.pth")
