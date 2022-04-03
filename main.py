import datetime
import xmltodict
from PIL import Image
from os import listdir

from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import DataLoader, random_split

WITH_MASK = "with_mask"
WITHOUT_MASK = "without_mask"
MASK_WEARED_INCORRECT = "mask_weared_incorrect"
OPTIONS = {WITH_MASK: 0, WITHOUT_MASK: 1, MASK_WEARED_INCORRECT: 2}
DEVICE = "cpu"

img_paths = list(sorted(listdir("./data/images/")))
lab_paths = list(sorted(listdir("./data/annotations/")))


def get_paths(path):
    paths = []
    for filename in listdir(path):
        paths.append(path + filename)
    return sorted(paths)


def get_metadata_paths():
    return get_paths("./data/annotations/")


def get_images_paths():
    return get_paths("./data/images/")


# Loop all annotation files and save mask data to list
def extract_metadata():
    data = []
    for lab_path in lab_paths:
        with open(lab_path) as file:
            # Parse to dict and select the 'object' element(s)
            metadata = xmltodict.parse(file.read())
            obj = metadata["annotation"]["object"]

            # If the object is a list loop through it, else select name
            if type(obj) == list:
                for o in obj:
                    data.append(o["name"])
            else:
                data.append(obj["name"])
    return data


def plot_mask_data():
    data = extract_metadata()
    x = ["With Mask", "Without Mask", "Mask Weared Incorrect"]
    y = [
        data.count(WITH_MASK),
        data.count(WITHOUT_MASK),
        data.count(MASK_WEARED_INCORRECT),
    ]

    _, ax = plt.subplots()
    ax.bar(x, y)

    plt.savefig("images/mask_data.png")


# Plot data and save diagram to file
# plot_mask_data()

# Dataset
def create_dataset():
    # Get file paths
    images_paths = get_images_paths()
    metadata_paths = get_metadata_paths()

    # Empty tensor lists
    image_tensor = []
    label_tensor = []

    # Composes several transforms together to be used on image
    t = transforms.Compose([transforms.Resize((226, 226)), transforms.ToTensor()])

    for i, image_path in enumerate(images_paths):

        # Debug logging
        print(f"tensor {i}")
        print(image_path)
        print(metadata_paths[i])

        with open(metadata_paths[i]) as file:
            metadata = xmltodict.parse(file.read())
            obj = metadata["annotation"]["object"]
            if type(obj) == list:
                # Loop through each object (face) in image
                for i in range(len(obj)):
                    # map xmin, ymin, xmax and ymax xml properties
                    xmin, ymin, xmax, ymax = list(map(int, obj[i]["bndbox"].values()))

                    # Get image label and convert it to OPTION number
                    label = OPTIONS[obj[i]["name"]]

                    # Crop the image based on xmin, ymin, xmax and ymax (face coordinates)
                    image = transforms.functional.crop(
                        Image.open(image_path).convert("RGB"),
                        ymin,
                        xmin,
                        ymax - ymin,
                        xmax - xmin,
                    )

                    # Apply transforms to image and append image to tensor list
                    image_tensor.append(t(image))

                    # append label to tensor list
                    label_tensor.append(torch.tensor(label))
            else:
                # map xmin, ymin, xmax, ymax xml properties
                x, y, w, h = list(map(int, obj["bndbox"].values()))

                # Get image label and convert it to OPTION number
                label = OPTIONS[obj["name"]]

                # Crop the image based on xmin, ymin, xmax and ymax (face coordinates)
                image = transforms.functional.crop(
                    Image.open(image_path).convert("RGB"), y, x, h - y, w - x
                )

                # Apply transforms to image and append image to tensor list
                image_tensor.append(t(image))

                # append label to tensor list
                label_tensor.append(torch.tensor(label))

    # Zip image and label tensor lists
    dataset = [[j, k] for j, k in zip(image_tensor, label_tensor)]
    return tuple(dataset)


# Create dataset
dataset = create_dataset()


def get_dataloaders():
    train_size = int(len(dataset) * 0.7)
    test_size = int(len(dataset) - train_size)
    train_set, test_set = random_split(dataset, [train_size, test_size])

    train_dl = DataLoader(dataset=train_set, batch_size=32, shuffle=True, num_workers=4)
    test_dl = DataLoader(dataset=test_set, batch_size=32, shuffle=True, num_workers=4)

    return train_dl, test_dl


# Model ResNet34
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = (X.to(DEVICE), y.to(DEVICE))

        optimizer.zero_grad()

        pred = model(X)

        loss = loss_fn(pred, y)
        loss.backward()

        optimizer.step()

        if batch % 20 == 10:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )


train_dataloader, test_dataloader = get_dataloaders()

model = models.resnet34(pretrained=True)
for param in model.parameters():
    param.requires_grad = False

n_inputs = model.fc.in_features
last_layer = nn.Linear(n_inputs, 3)

model.fc.out_features = last_layer

# conv_param = 64 * 128 * 3 * 3
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# We need this because otherwise loss.backward() doesnt work
param.requires_grad = True
ct = 0
for child in model.children():
    ct += 1
    if ct < 7:
        for param in child.parameters():
            param.requires_grad = False

epochs = 1
for t in range(epochs):
    ct = datetime.datetime.now()
    print(f"{ct} - Epoch {t+1}\n")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Finished!")


# Training Model


# for epoch in range(1, 2):
#     running_loss = 0.0
#     train_losses = []

#     for i, (inputs, labels) in enumerate(get_train_dataloader()):
#         inputs = inputs.to("cpu")
#         labels = labels.to("cpu")

#         optimizer.zero_grad()

#         outputs = model(inputs)
#         loss = loss_fn(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item()
#         if i % 20 == 19:
#             print(
#                 "Epoch {}, batch {}, training loss {}".format(
#                     epoch, i + 1, running_loss / 20
#                 )
#             )

#         running_loss = 0.0
