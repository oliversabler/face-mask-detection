import random
import xmltodict
import cv2
import time

from PIL import Image
from os import listdir, path
from datetime import datetime

from matplotlib import pyplot as plt
import numpy as np

import torch
import torchvision.transforms as transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import Dataset, DataLoader

WITH_MASK = "with_mask"
WITHOUT_MASK = "without_mask"
MASK_WEARED_INCORRECT = "mask_weared_incorrect"

img_paths = "./data/images/"
xml_paths = "./data/annotations/"
filenames = listdir(img_paths)

"""
    Visualization Helper Methods
"""


def get_label_id(label):
    if label == WITH_MASK:
        label_id = 2
    elif label == WITHOUT_MASK:
        label_id = 1
    elif label == MASK_WEARED_INCORRECT:
        label_id = 0
    return label_id


def get_box_color(label):
    if label == 2:
        color = (0, 255, 0)
    elif label == 1:
        color = (0, 0, 255)
    else:
        color = (255, 0, 0)
    return color


"""
    Visualization
"""


def get_annotation(filename):
    bndboxes = []
    labels = []

    xml_path = path.join(xml_paths, filename[:-3] + "xml")

    with open(xml_path) as file:
        xml = xmltodict.parse(file.read())
        root = xml["annotation"]

        obj = root["object"]
        if type(obj) != list:
            obj = [obj]

        for obj in obj:
            xmin, ymin, xmax, ymax = list(map(int, obj["bndbox"].values()))
            bndboxes.append([xmin, ymin, xmax, ymax])

            label = obj["name"]
            labels.append(get_label_id(label))

        return bndboxes, labels


def mark_faces(img, bndboxes, labels):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    for bndbox, label in zip(bndboxes, labels):
        cv2.rectangle(
            img,
            (bndbox[0], bndbox[1]),
            (bndbox[2], bndbox[3]),
            color=get_box_color(label),
            thickness=1,
        )
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def visualize_random_image():
    image_name = filenames[random.randint(0, len(filenames))]
    bndboxes, labels = get_annotation(image_name)
    img_path = path.join(img_paths, image_name)

    img = mark_faces(plt.imread(img_path), bndboxes, labels)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    plt.axis("off")
    ax.legend(title=image_name)
    ax.imshow(img)
    fig.savefig("test.png")


"""
    Mask Dataset    
"""


class MaskDataset(Dataset):
    def __init__(self, filenames, img_paths):
        self.filenames = filenames
        self.img_paths = img_paths

    def __getitem__(self, index):
        img_name = self.filenames[index]
        img_path = path.join(self.img_paths, img_name)
        img = Image.open(img_path).convert("RGB")
        img = transforms.ToTensor()(img)

        bndboxes, labels = get_annotation(img_name)
        boxes = torch.as_tensor(bndboxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        area = torch.as_tensor(area, dtype=torch.float32)
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

        target = {}

        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([index])
        target["area"] = area
        target["iscrowed"] = iscrowd
        return img, target

    def __len__(self):
        return len(self.filenames)


"""
    Data Loader
"""


def collate_fn(batch):
    return tuple(zip(*batch))


def get_dataloader():
    mask_dataset = MaskDataset(filenames, img_paths)

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
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.SGD(params, lr=0.01, momentum=0.9, weight_decay=0.0005)


def train():
    dataloader = get_dataloader()
    model = get_model()
    optimizer = get_optimizer(model)

    epochs = 5
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


"""
    Testing
"""
# visualize_random_image()
train()
