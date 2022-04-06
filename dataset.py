from PIL import Image
from os import path

import torch
from torch.utils.data import Dataset, DataLoader

from globals import FILENAMES, IMGS_PATH
from utils import get_annotation

import pytorch_helpers.transforms as T


class MaskDataset(Dataset):
    def __init__(self, transforms=None):
        super().__init__()

        self.transforms = transforms

    def __getitem__(self, index):
        img_name = FILENAMES[index]
        img_path = path.join(IMGS_PATH, img_name)
        img = Image.open(img_path).convert("RGB")

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
        if self.transforms is not None:
            img, target = self.transforms(img, target)

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


# Todo: Move?
#       Divide dataset to training (~70%) and testing (~30%)
def get_dataloader():
    mask_dataset = MaskDataset(get_transform(train=True))
    print("test")
    return DataLoader(
        mask_dataset,
        batch_size=5,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
    )
