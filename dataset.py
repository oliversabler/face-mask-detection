from os import path
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from globals import FILENAMES, IMGS_PATH
from utils import get_annotation


class MaskDataset(Dataset):
    def __init__(self):
        super().__init__()

    def __getitem__(self, index):
        img_name = FILENAMES[index]
        img_path = path.join(IMGS_PATH, img_name)

        trans = transforms.Compose([transforms.ToTensor()])
        img = Image.open(img_path).convert("RGB")
        img = trans(img)

        w, h = img.shape[1], img.shape[0]
        boxes, labels = get_annotation(img_name, w, h)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        area = torch.as_tensor(area, dtype=torch.float32)
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

        """
        - image_id (Int64Tensor[1]): an image identifier. It should be unique between all the images in the dataset, and is used during evaluation.
        - boxes (FloatTensor[N, 4]): the coordinates of the N bounding boxes in [x0, y0, x1, y1] format, ranging from 0 to W and 0 to H.
        - labels (Int64Tensor[N]): the label for each bounding box. 0 represents always the background class.
        - area (Tensor[N]): The area of the bounding box. This is used during evaluation with the COCO metric, to separate the metric scores between small, medium and large boxes.
        - iscrowd (UInt8Tensor[N]): instances with iscrowd=True will be ignored during evaluation.
        """
        target = {}
        target["image_id"] = torch.tensor([index])
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = area
        target["iscrowd"] = iscrowd

        return img, target

    def __len__(self):
        return len(FILENAMES)
