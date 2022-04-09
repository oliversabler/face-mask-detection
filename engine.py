import math
import sys
import time

from datetime import datetime

import torch
from globals import DEVICE
from model import get_resnet50_model, get_sgd_optimizer, get_dataloader

from pytorch_helpers.engine import evaluate, train_one_epoch


def _warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def _log_iter(epoch, i, loss_value, time, lr):
    print(
        "Epoch [{}] Iteration [{}] - Loss: {:.4f}, Time: {:.4f}, Learning Rate: {:.6f}".format(
            epoch,
            i,
            loss_value,
            time,
            lr,
        )
    )


def _log_iter_eval(epoch, i, prediction, truth, time):
    print(
        "Epoch [{}] Iteration [{}] Evaluation - Prediction no. boxes: {}, Truth no. boxes: {}, Time: {:.4f}".format(
            epoch, i, prediction, truth, time
        )
    )


def _train_epoch(model, optimizer, dataloader, epoch):
    model.train()

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(dataloader) - 1)

        lr_scheduler = _warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    i = 0

    for images, targets in dataloader:
        time_start = time.time()

        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        _log_iter(
            epoch,
            i,
            loss_value,
            time.time() - time_start,
            optimizer.param_groups[0]["lr"],
        )

        i += 1

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)


def _evaluate_epoch(model, dataloader, epoch):
    model.eval()

    with torch.no_grad():
        i = 0
        for images, targets in dataloader:
            time_start = time.time()

            images = list(image.to(DEVICE) for image in images)
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            predictions = model(images)

            _log_iter_eval(
                epoch,
                i,
                len(predictions[i]["labels"]),
                len(targets[i]["labels"]),
                time.time() - time_start,
            )

            i += 1


def train():
    print("[Training]")

    print(f"Using device: {DEVICE}")

    model = get_resnet50_model()
    optimizer = get_sgd_optimizer(model)
    dataloader, dataloader_test = get_dataloader(take_one=True)

    epochs = 1
    print(f"Number of epochs: {epochs}")

    print("Starting...")
    for epoch in range(epochs):
        _train_epoch(model, optimizer, dataloader, epoch)
        _evaluate_epoch(model, dataloader_test, epoch)

        # PyTorch
        # train_one_epoch(model, optimizer, dataloader, DEVICE, epoch, print_freq=1)
        # evaluate(model, dataloader_test, DEVICE)

    torch.save(model.state_dict(), f"./models/model.pth")
    torch.save(model.state_dict(), f"./models/model_{datetime.now()}.pth")
