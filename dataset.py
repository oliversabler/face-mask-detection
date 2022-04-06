import cv2
from os import path

import numpy as np

import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from globals import FILENAMES, IMGS_PATH
from utils import get_annotation

import pytorch_helpers.transforms as T


class MaskDataset(Dataset):
    def __init__(self, transforms=None):
        super().__init__()

        self.transforms = transforms

    def __getitem__(self, index):
        # Read image name by index and construct path
        img_name = FILENAMES[index]
        img_path = path.join(IMGS_PATH, img_name)

        # Open image
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img = cv2.resize(img, (480, 480), cv2.INTER_AREA)
        img /= 255.0

        # Read annotation file linked to image to get boxes and labels
        # Convert to Tensors
        boxes, labels = get_annotation(img_name)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        area = torch.as_tensor(area, dtype=torch.float32)
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

        target = {}

        # boxes (FloatTensor[N, 4]): the coordinates of the N bounding boxes in [x0, y0, x1, y1] format, ranging from 0 to W and 0 to H.
        target["boxes"] = boxes

        # labels (Int64Tensor[N]): the label for each bounding box. 0 represents always the background class.
        target["labels"] = labels

        # image_id (Int64Tensor[1]): an image identifier. It should be unique between all the images in the dataset, and is used during evaluation.
        target["image_id"] = torch.tensor([index])

        # area (Tensor[N]): The area of the bounding box. This is used during evaluation with the COCO metric, to separate the metric scores between small, medium and large boxes.
        target["area"] = area

        # iscrowd (UInt8Tensor[N]): instances with iscrowd=True will be ignored during evaluation.
        target["iscrowd"] = iscrowd

        img = transforms.ToTensor()(img)
        # Run transforms on image and target
        # if self.transforms is not None:
        #    img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(FILENAMES)


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def collate_fn(batch):
    return tuple(zip(*batch))


def get_dataloader():
    mask_dataset = MaskDataset(get_transform(train=True))
    mask_dataset_test = MaskDataset(get_transform(train=False))

    indices = torch.randperm(len(mask_dataset)).tolist()
    mask_dataset = torch.utils.data.Subset(mask_dataset, indices[:-150])
    mask_dataset_test = torch.utils.data.Subset(mask_dataset_test, indices[-150:])

    train_dataset = DataLoader(
        mask_dataset,
        batch_size=20,
        shuffle=True,
        num_workers=4,
        drop_last=True,
        persistent_workers=True,
        collate_fn=collate_fn,
    )

    test_dataset = DataLoader(
        mask_dataset_test,
        batch_size=1,
        shuffle=True,
        num_workers=4,
        drop_last=True,
        persistent_workers=True,
        collate_fn=collate_fn,
    )

    return train_dataset, test_dataset
