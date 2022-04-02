import enum
import xmltodict
import numpy as np
from matplotlib import pyplot as plt
from os import listdir

WITH_MASK = "with_mask"
WITHOUT_MASK = "without_mask"
MASK_WEARED_INCORRECT = "mask_weared_incorrect"
OPTIONS = {WITH_MASK: 0, WITHOUT_MASK: 1, MASK_WEARED_INCORRECT: 2}


def get_paths(path):
    paths = []
    for filename in listdir(path):
        paths.append(path + filename)
    return paths


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

# Process images (mark faces)?

# Dataset
def handle_obj_list(obj):
    for i in range(len(obj)):
        x, y, w, h = list(map(int, obj[i]["bndbox"].values()))
        label = OPTIONS[obj[i]["name"]]


def handle_obj(obj):
    x, y, w, h = list(map(int, obj["bndbox"].values()))
    label = OPTIONS[obj["name"]]
    print(label)


def create_dataset():
    images_paths = get_images_paths()
    metadata_paths = get_metadata_paths()
    image_tensor = []
    label_tensor = []

    data_len = int(len(images_paths)) - 1
    for i in range(data_len):
        with open(metadata_paths[i]) as file:
            metadata = xmltodict.parse(file.read())
            obj = metadata["annotation"]["object"]
            if type(obj) == list:
                handle_obj_list(obj)
            else:
                handle_obj(obj)


create_dataset()
