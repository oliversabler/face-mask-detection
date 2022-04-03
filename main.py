import time
import random
import xmltodict
import cv2

from collections import OrderedDict
from PIL import Image
from os import listdir, path

from matplotlib import pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import Dataset, DataLoader, random_split

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


visualize_random_image()

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

        bndbox, labels = get_annotation(img_name)
        bndboxes = torch.as_tensor(bndbox, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        area = (bndboxes[:, 2] - bndboxes[:, 0]) * (bndboxes[:, 3] - bndboxes[:, 1])
        area = torch.as_tensor(area, dtype=torch.float32)
        iscrowd = torch.zeros((len(bndbox),), dtype=torch.int64)

        target = {}

        target["boxes"] = bndbox
        target["labels"] = labels
        target["image_id"] = torch.tensor([index])
        target["area"] = area
        target["iscrowed"] = iscrowd
        return img, target

    def __len__(self):
        return len(self.filenames)


mask_dataset = MaskDataset(filenames, img_paths)
