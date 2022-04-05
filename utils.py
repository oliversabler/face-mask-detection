import xmltodict
import cv2

from PIL import Image
from os import path

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


WITH_MASK = "with_mask"
WITHOUT_MASK = "without_mask"
MASK_WEARED_INCORRECT = "mask_weared_incorrect"


class MaskDataset(Dataset):
    def __init__(self, filenames, imgs_path, xmls_path):
        self.filenames = filenames
        self.img_paths = imgs_path
        self.xmls_path = xmls_path

    def __getitem__(self, index):
        xmls_path = self.xmls_path
        img_name = self.filenames[index]
        img_path = path.join(self.img_paths, img_name)
        img = Image.open(img_path).convert("RGB")
        img = transforms.ToTensor()(img)

        bndboxes, labels = get_annotation(img_name, xmls_path)
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


def collate_fn(batch):
    return tuple(zip(*batch))


def get_dataloader(filenames, imgs_path, xmls_path):
    mask_dataset = MaskDataset(filenames, imgs_path, xmls_path)

    return DataLoader(
        mask_dataset, batch_size=5, shuffle=True, num_workers=4, collate_fn=collate_fn
    )


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


def get_annotation(filename, xmls_path):
    bndboxes = []
    labels = []

    xml_path = path.join(xmls_path, filename[:-3] + "xml")

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
            (int(bndbox[0]), int(bndbox[1])),
            (int(bndbox[2]), int(bndbox[3])),
            color=get_box_color(label),
            thickness=1,
        )
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
