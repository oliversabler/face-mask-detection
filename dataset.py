"""
Handles the dataset
"""
import os
from PIL import Image

import torch
from torch.utils.data import Dataset

from utils import get_annotation

class MaskDataset(Dataset):
    """
    - image_id: Image identifier.
    - boxes: The coordinates of the N bounding boxes in [x0, y0, x1, y1] format
    - labels: The label for each bounding box. 0 represents always the background class.
    - area: The area of the bounding box.
    - iscrowd: Instances with iscrowd=True will be ignored during evaluation.
    """
    def __init__(self, filenames, imgs_path, xmls_path, transforms):
        self.filenames = filenames
        self.imgs_path = imgs_path
        self.xmls_path = xmls_path
        self.transforms = transforms

    def __getitem__(self, index):
        img_name = self.filenames[index]
        img_path = os.path.join(self.imgs_path, img_name)
        img = Image.open(img_path).convert('RGB')
        width, height = img.size
        boxes, labels = get_annotation(img_name, self.xmls_path, width, height)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        area = torch.as_tensor(area, dtype=torch.float32)
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

        target = {}
        target['image_id'] = torch.tensor([index])
        target['boxes'] = boxes
        target['labels'] = labels
        target['area'] = area
        target['iscrowd'] = iscrowd

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.filenames)
