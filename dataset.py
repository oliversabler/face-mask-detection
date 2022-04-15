import cv2
from os import path

import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

from globals import FILENAMES, IMGS_PATH
from utils import get_annotation, mark_faces


class MaskDataset(Dataset):
    def __init__(self):
        super().__init__()

    def __getitem__(self, index):
        img_name = FILENAMES[index]
        img_path = path.join(IMGS_PATH, img_name)

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)

        w, h = img.shape[1], img.shape[0]
        w_r, h_r = int(w * 0.75), int(h * 0.75)
        print(w_r, h_r)
        img = cv2.resize(img, (w_r, h_r), cv2.INTER_AREA)
        img /= 255.0

        boxes, labels = get_annotation(img_name, w, h, w_r, h_r)

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

        # Testing
        from matplotlib import pyplot as plt

        img = mark_faces(img, boxes, labels)
        fig, ax = plt.subplots(1, 1)
        plt.axis("off")
        ax.legend(title=img_name)
        ax.imshow(img)
        fig.savefig("./temp/visualization.png", bbox_inches="tight", pad_inches=0)

        # Todo: Transforms
        # https://stackoverflow.com/questions/63068332/does-pytorch-allow-to-apply-given-transformations-to-bounding-box-coordinates-of

        img = T.ToTensor()(img)

        return img, target

    def __len__(self):
        return len(FILENAMES)
