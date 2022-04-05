import time
from datetime import datetime

import torch
import numpy as np

from models import get_resnet50_model
from optimizers import get_sgd_optimizer
from utils import get_dataloader

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Device: {device}")


def train(filenames, imgs_path, xmls_path):
    dataloader = get_dataloader(filenames, imgs_path, xmls_path)
    model = get_resnet50_model()
    optimizer = get_sgd_optimizer(model)

    epochs = 3
    loss_list = []

    for epoch in range(epochs):
        print(f"Starting traning (epoch {epoch + 1}/{epochs}) --- {datetime.now()}")
        loss_sub_list = []
        for batch, (images, targets) in enumerate(dataloader):
            start_time = time.time()

            print(f"Batch #{batch + 1} | Batch size: {len(images)}")

            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            model.train()
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()
            loss_sub_list.append(loss_value)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            end_time = time.time()
            print(
                "#{} Loss: {:.3f} ({:.1f}s)".format(
                    batch + 1, loss_value, end_time - start_time
                )
            )

        epoch_loss = np.mean(loss_sub_list)
        loss_list.append(epoch_loss)
        print(f"{datetime.now()} | Epoch loss: {epoch_loss}")

    torch.save(model.state_dict(), f"./models/model.pth")
    torch.save(model.state_dict(), f"./models/model_{datetime.now()}.pth")
