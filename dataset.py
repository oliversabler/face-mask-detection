from PIL import Image
from os import path

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from utils import get_annotation


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
