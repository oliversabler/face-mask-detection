from datetime import datetime

import torch
from globals import DEVICE
from model import get_resnet50_model
from optims import get_sgd_optimizer
from dataset import get_dataloader

from pytorch_helpers.engine import train_one_epoch


def train():
    print("[Training]")

    print(f"Using device: {DEVICE}")

    dataloader = get_dataloader()
    model = get_resnet50_model()
    optimizer = get_sgd_optimizer(model)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    epochs = 5
    print(f"Number of epochs: {epochs}")

    print("Starting...")
    for epoch in range(epochs):
        train_one_epoch(model, optimizer, dataloader, DEVICE, epoch, print_freq=1)
        lr_scheduler.step()

    torch.save(model.state_dict(), f"./models/model.pth")
    torch.save(model.state_dict(), f"./models/model_{datetime.now()}.pth")
