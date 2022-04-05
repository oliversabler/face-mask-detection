import random
import time

from PIL import Image
from os import listdir, path
from datetime import datetime

from matplotlib import pyplot as plt
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader

from dataset import MaskDataset
from utils import get_annotation, mark_faces


imgs_path = "./data/images/"
xmls_path = "./data/annotations/"
filenames = listdir(imgs_path)


def visualize_random_image():
    image_name = filenames[random.randint(0, len(filenames))]
    bndboxes, labels = get_annotation(image_name, xmls_path)
    img_path = path.join(imgs_path, image_name)

    img = mark_faces(plt.imread(img_path), bndboxes, labels)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    plt.axis("off")
    ax.legend(title=image_name)
    ax.imshow(img)
    fig.savefig("test.png")


"""
    Data Loader
"""


def collate_fn(batch):
    return tuple(zip(*batch))


def get_dataloader():
    mask_dataset = MaskDataset(filenames, imgs_path, xmls_path)

    return DataLoader(
        mask_dataset, batch_size=5, shuffle=True, num_workers=4, collate_fn=collate_fn
    )


"""
    Model
"""

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Device: {device}")


def get_model():
    num_classes = 4

    model = fasterrcnn_resnet50_fpn(pretrained=True)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    model = model.to(device)

    return model


def get_optimizer(model):
    # Adam optimizer
    # for param in model.parameters():
    #     param.grad = None

    # return torch.optim.Adam(model.parameters(), lr=0.01)

    # SGD optimizer
    params = [p for p in model.parameters() if p.requires_grad]

    return torch.optim.SGD(params, lr=0.01, momentum=0.9, weight_decay=0.0005)


def train():
    dataloader = get_dataloader()
    model = get_model()
    optimizer = get_optimizer(model)

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


"""
    Prediction
"""


def predict_img(img, nm_thrs=0.3, score_thrs=0.8):
    test_img = transforms.ToTensor()(img)

    model = get_model()
    model.load_state_dict(torch.load("./models/model_2022-04-05 05:05:46.048959.pth"))
    model.eval()

    with torch.no_grad():
        predictions = model(test_img.unsqueeze(0).to(device))

    test_img = test_img.permute(1, 2, 0).numpy()
    keep_boxes = torchvision.ops.nms(
        predictions[0]["boxes"].cpu(), predictions[0]["scores"].cpu(), nm_thrs
    )

    score_filter = predictions[0]["scores"].cpu().numpy()[keep_boxes] > score_thrs

    test_boxes = predictions[0]["boxes"].cpu().numpy()[keep_boxes][score_filter]
    test_labels = predictions[0]["labels"].cpu().numpy()[keep_boxes][score_filter]

    return test_img, test_boxes, test_labels


def predict_random_image():
    random_image_name = filenames[random.randint(0, len(filenames))]
    test_img = Image.open(path.join(imgs_path, random_image_name)).convert("RGB")

    # Prediction
    test_img, test_boxes, test_labels = predict_img(test_img)
    test_output = mark_faces(test_img, test_boxes, test_labels)

    # Solution
    bndbox, labels = get_annotation(random_image_name, xmls_path)
    gt_output = mark_faces(test_img, bndbox, labels)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    ax1.imshow(test_output)
    ax1.set_xlabel("Prediction")
    ax2.imshow(gt_output)
    ax2.set_xlabel("Truth")
    plt.savefig("prediction.png")


"""
    Testing
"""
# visualize_random_image()
# train()
# predict_random_image()
