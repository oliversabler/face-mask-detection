import xmltodict
import numpy as np
from matplotlib import pyplot as plt
from os import listdir


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
        data.count("with_mask"),
        data.count("without_mask"),
        data.count("mask_weared_incorrect"),
    ]

    fig, ax = plt.subplots()
    ax.bar(x, y)

    plt.savefig("images/mask_data.png")


# Plot data and save diagram to file
plot_mask_data()

# Process images (mark faces)
