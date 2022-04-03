import datetime
from fileinput import filename
import random
import xmltodict
import cv2
from collections import OrderedDict
from PIL import Image
from os import listdir, path

from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader, random_split

WITH_MASK = "with_mask"
WITHOUT_MASK = "without_mask"
MASK_WEARED_INCORRECT = "mask_weared_incorrect"

img_paths = "./data/images/"
xml_paths = "./data/annotations/"
filenames = listdir(img_paths)

"""
    Visualization Helper Methods
"""


def get_label_id(label):
    if label == WITH_MASK:
        label_id = 2
    elif label == WITHOUT_MASK:
        label_id = 1
    elif label == MASK_WEARED_INCORRECT:
        label_id = 0
    return label_id


def get_box_color(label):
    if label == 2:
        color = (0, 255, 0)
    elif label == 1:
        color = (0, 0, 255)
    else:
        color = (255, 0, 0)
    return color


"""
    Visualization
"""


def get_annotation(filename):
    bndboxes = []
    labels = []

    xml_path = path.join(xml_paths, filename[:-3] + "xml")

    with open(xml_path) as file:
        xml = xmltodict.parse(file.read())
        root = xml["annotation"]

        obj = root["object"]
        if type(obj) != list:
            obj = [obj]

        for obj in obj:
            xmin, ymin, xmax, ymax = list(map(int, obj["bndbox"].values()))
            bndboxes.append([xmin, ymin, xmax, ymax])

            label = obj["name"]
            labels.append(get_label_id(label))

        return bndboxes, labels


def mark_faces(img, bndboxes, labels):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    for bndbox, label in zip(bndboxes, labels):
        cv2.rectangle(
            img,
            (bndbox[0], bndbox[1]),
            (bndbox[2], bndbox[3]),
            color=get_box_color(label),
            thickness=1,
        )
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def visualize_random_image():
    image_name = filenames[random.randint(0, len(filenames))]
    bndboxes, labels = get_annotation(image_name)
    img_path = path.join(img_paths, image_name)

    img = mark_faces(plt.imread(img_path), bndboxes, labels)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    plt.axis("off")
    ax.legend(title=image_name)
    ax.imshow(img)
    fig.savefig("test.png")


visualize_random_image()


# # Dataset
# class MaskDataset(Dataset):
#     def __init__(self, filenames, img_path, xml_path):
#         self.filenames = filenames
#         self.img_path = img_path
#         self.xml_path = xml_path

#     def __getitem__(self):
#         # Load image
#         img = Image.open(self.img_path).convert("RGB")
#         img = transforms.ToTensor()(img)

#         # Build target dict
#         bnbox, labels = get_annotations()

# # Dataset
# def create_dataset():
#     # Empty tensor lists
#     image_tensor = []
#     label_tensor = []

#     # Composes several transforms together to be used on image
#     t = transforms.Compose([transforms.Resize((226, 226)), transforms.ToTensor()])

#     for i, image_path in enumerate(lab_paths):

#         # Debug logging
#         print(f"tensor {i}")
#         print(image_path)
#         print(lab_paths[i])

#         with open(lab_paths[i]) as file:
#             metadata = xmltodict.parse(file.read())
#             obj = metadata["annotation"]["object"]
#             if type(obj) == list:
#                 # Loop through each object (face) in image
#                 for i in range(len(obj)):
#                     # map xmin, ymin, xmax and ymax xml properties
#                     xmin, ymin, xmax, ymax = list(map(int, obj[i]["bndbox"].values()))

#                     # Get image label and convert it to OPTION number
#                     label = OPTIONS[obj[i]["name"]]

#                     # Crop the image based on xmin, ymin, xmax and ymax (face coordinates)
#                     image = transforms.functional.crop(
#                         Image.open(image_path).convert("RGB"),
#                         ymin,
#                         xmin,
#                         ymax - ymin,
#                         xmax - xmin,
#                     )

#                     # Apply transforms to image and append image to tensor list
#                     image_tensor.append(t(image))

#                     # append label to tensor list
#                     label_tensor.append(torch.tensor(label))
#             else:
#                 # map xmin, ymin, xmax, ymax xml properties
#                 x, y, w, h = list(map(int, obj["bndbox"].values()))

#                 # Get image label and convert it to OPTION number
#                 label = OPTIONS[obj["name"]]

#                 # Crop the image based on xmin, ymin, xmax and ymax (face coordinates)
#                 image = transforms.functional.crop(
#                     Image.open(image_path).convert("RGB"), y, x, h - y, w - x
#                 )

#                 # Apply transforms to image and append image to tensor list
#                 image_tensor.append(t(image))

#                 # append label to tensor list
#                 label_tensor.append(torch.tensor(label))

#     # Zip image and label tensor lists
#     dataset = [[j, k] for j, k in zip(image_tensor, label_tensor)]
#     return tuple(dataset)


# # Create dataset
# dataset = create_dataset()


# def get_dataloaders():
#     train_size = int(len(dataset) * 0.7)
#     test_size = int(len(dataset) - train_size)
#     train_set, test_set = random_split(dataset, [train_size, test_size])

#     train_dl = DataLoader(dataset=train_set, batch_size=32, shuffle=True, num_workers=4)
#     test_dl = DataLoader(dataset=test_set, batch_size=32, shuffle=True, num_workers=4)

#     return train_dl, test_dl


# # Model ResNet34
# def train(dataloader, model, loss_fn, optimizer):
#     size = len(dataloader.dataset)
#     # model.train()
#     for batch, (X, y) in enumerate(dataloader):
#         X, y = (X.to(DEVICE), y.to(DEVICE))

#         optimizer.zero_grad()

#         pred = model(X)

#         loss = loss_fn(pred, y)
#         loss.backward()

#         optimizer.step()

#         if batch % 20 == 10:
#             loss, current = loss.item(), batch * len(X)
#             print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")


# def test(dataloader, model, loss_fn):
#     size = len(dataloader.dataset)
#     num_batches = len(dataloader)
#     model.eval()
#     test_loss, correct = 0, 0
#     with torch.no_grad():
#         for X, y in dataloader:
#             X, y = X.to(DEVICE), y.to(DEVICE)
#             pred = model(X)
#             test_loss += loss_fn(pred, y).item()
#             correct += (pred.argmax(1) == y).type(torch.float).sum().item()
#     test_loss /= num_batches
#     correct /= size
#     print(
#         f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
#     )


# train_dataloader, test_dataloader = get_dataloaders()

# model = models.resnet34(pretrained=True)
# for param in model.parameters():
#     param.requires_grad = False

# n_inputs = model.fc.in_features
# last_layer = nn.Linear(n_inputs, 3)

# model.fc.out_features = last_layer

# # conv_param = 64 * 128 * 3 * 3
# loss_fn = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# # We need this because otherwise loss.backward() doesnt work
# param.requires_grad = True
# ct = 0
# for child in model.children():
#     ct += 1
#     if ct < 7:
#         for param in child.parameters():
#             param.requires_grad = False

# epochs = 1
# for t in range(epochs):
#     ct = datetime.datetime.now()
#     print(f"{ct} - Epoch {t+1}\n")
#     train(train_dataloader, model, loss_fn, optimizer)
#     test(test_dataloader, model, loss_fn)
# print("Finished!")
