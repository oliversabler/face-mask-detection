import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from globals import DEVICE


def get_resnet50_model():
    model = fasterrcnn_resnet50_fpn(pretrained=True)

    num_classes = 3
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    model = model.to(DEVICE)

    return model


def load_resnet50_model_state(path):
    model = get_resnet50_model()
    model.load_state_dict(torch.load(path))
    return model
