import random
import cv2

from os import path

from matplotlib import pyplot as plt
import numpy as np

from utils import get_annotation, plot_image
from globals import FILENAMES, IMGS_PATH


def _visualize_image(img_name):
    img_path = path.join(IMGS_PATH, img_name)

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    img = cv2.resize(img, (480, 480), cv2.INTER_AREA)
    img /= 255.0

    bboxes, labels = get_annotation(img_name)
    img_path = path.join(IMGS_PATH, img_name)

    img = plot_image(plt.imread(img_path), bboxes, labels)

    fig, ax = plt.subplots(1, 1)
    plt.axis("off")
    ax.legend(title=img_name)
    ax.imshow(img)
    fig.savefig("./temp/visualization.png", bbox_inches="tight", pad_inches=0)


def visualize_random_image():
    img_name = FILENAMES[random.randint(0, len(FILENAMES))]
    _visualize_image(img_name)


def visualize_image_by_index(index=0):
    img_name = FILENAMES[index]
    _visualize_image(img_name)
