"""
Handles the model and optimizer
"""
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def get_resnet50_model(device):
    """
    Get the Faster RCNN ResNet 50 model
    """
    # Need this to download model, not sure if this is the right place
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context

    # Fetch model
    resnet = fasterrcnn_resnet50_fpn(weights='FasterRCNN_ResNet50_FPN_Weights.DEFAULT')

    # Classes: Background, mask, mask wrong, no mask
    num_classes = 4

    # Not sure what this do
    in_features = resnet.roi_heads.box_predictor.cls_score.in_features
    resnet.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    resnet.to(device)

    # Evaluate the model
    resnet.eval()

    return resnet

def load_resnet50_model_state(path, device):
    """
    Load the model for prediction
    """
    model = get_resnet50_model()
    model.load_state_dict(torch.load(path, map_location=torch.device(device)))
    return model

def get_sgd_optimizer(model):
    """
    Get the SGD Optimizer
    """
    params = [p for p in model.parameters() if p.requires_grad]

    return torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

def _collate_fn(batch):
    return tuple(zip(*batch))

def get_dataloader(dataset, batch_size=1, take_one=False):
    """
    Get the dataloaders for training end evalutation
    """
    #mask_dataset = MaskDataset(_get_transform(train=True))
    #mask_dataset_test = MaskDataset(_get_transform())

    indices = torch.randperm(len(dataset)).tolist()

    if take_one:
        dataset = Subset(dataset, indices[:1])
        #mask_dataset_test = Subset(mask_dataset_test, indices[:1])
    else:
        dataset = Subset(dataset, indices[:-100])
        #mask_dataset_test = Subset(mask_dataset_test, indices[-100:])

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=_collate_fn
    )

    return dataloader
