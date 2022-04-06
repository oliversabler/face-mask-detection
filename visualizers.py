import random
from os import path
from matplotlib import pyplot as plt
from utils import get_annotation, mark_faces

from globals import FILENAMES, IMGS_PATH, XMLS_PATH


def visualize_random_image():
    image_name = FILENAMES[random.randint(0, len(FILENAMES))]
    bboxes, labels = get_annotation(image_name, XMLS_PATH)
    img_path = path.join(IMGS_PATH, image_name)

    img = mark_faces(plt.imread(img_path), bboxes, labels)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    plt.axis("off")
    ax.legend(title=image_name)
    ax.imshow(img)
    fig.savefig("test.png")
