import torch
from torch.utils.data import DataLoader, Subset
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from dataset import MaskDataset
from globals import DEVICE


def get_resnet50_model():
    model = fasterrcnn_resnet50_fpn(pretrained=True)

    num_classes = 4  # do I need 5?
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    model = model.to(DEVICE)

    return model


def load_resnet50_model_state(path):
    model = get_resnet50_model()
    model.load_state_dict(torch.load(path))
    return model


def get_sgd_optimizer(model):
    params = [p for p in model.parameters() if p.requires_grad]

    return torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)


def _collate_fn(batch):
    return tuple(zip(*batch))


def get_dataloader(train_batch_size=1, test_batch_size=1, take_one=False):
    mask_dataset = MaskDataset()
    mask_dataset_test = MaskDataset()

    indices = torch.randperm(len(mask_dataset)).tolist()

    if take_one:
        mask_dataset = Subset(mask_dataset, indices[:1])
        mask_dataset_test = Subset(mask_dataset_test, indices[:1])
    else:
        mask_dataset = Subset(mask_dataset, indices[:-100])
        mask_dataset_test = Subset(mask_dataset_test, indices[-100:])

    train_dataset = DataLoader(
        mask_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=1,
        drop_last=True,
        persistent_workers=True,
        collate_fn=_collate_fn,
    )

    test_dataset = DataLoader(
        mask_dataset_test,
        batch_size=test_batch_size,
        shuffle=True,
        num_workers=1,
        drop_last=True,
        persistent_workers=True,
        collate_fn=_collate_fn,
    )

    return train_dataset, test_dataset
