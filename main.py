import xmltodict
from PIL import Image
from os import listdir

import numpy as np
from matplotlib import pyplot as plt

import torch
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader, random_split

WITH_MASK = "with_mask"
WITHOUT_MASK = "without_mask"
MASK_WEARED_INCORRECT = "mask_weared_incorrect"
OPTIONS = {WITH_MASK: 0, WITHOUT_MASK: 1, MASK_WEARED_INCORRECT: 2}


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
    for path in get_metadata_paths():
        with open(path) as file:
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

    fig, ax = plt.subplots()
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
                    x, y, w, h = list(map(int, obj[i]["bndbox"].values()))

                    # Get image label and convert it to OPTION number
                    label = OPTIONS[obj[i]["name"]]

                    # Crop the image based on xmin, ymin, xmax and ymax (face coordinates)
                    image = transforms.functional.crop(
                        Image.open(image_path).convert("RGB"), y, x, h - y, w - x
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
ds = create_dataset()


def train_samples():
    train_size = int(len(ds) * 0.7)
    test_size = int(len(ds) - train_size)
    train_set, _ = random_split(ds, [train_size, test_size])

    train_dl = DataLoader(
        dataset=train_set, batch_size=32, shuffle=False, num_workers=4
    )

    train_features, _ = next(iter(train_dl))

    train_features_np = train_features.numpy()

    for i in np.arange(3):
        plt.imsave(f"test{i}.png", np.transpose(train_features_np[i], (1, 2, 0)))


train_samples()
