import torch


def get_sgd_optimizer(model):
    params = [p for p in model.parameters() if p.requires_grad]

    return torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)


def get_adam_optimizer(model):
    # Absolutely not working at the moment
    params = [p for p in model.parameters() if p.requires_grad]

    return torch.optim.Adam(
        model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005
    )