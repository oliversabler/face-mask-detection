import random
from os import path
from matplotlib import pyplot as plt
from utils import get_annotation, mark_faces


def visualize_random_image(filenames, imgs_path, xmls_path):
    image_name = filenames[random.randint(0, len(filenames))]
    bndboxes, labels = get_annotation(image_name, xmls_path)
    img_path = path.join(imgs_path, image_name)

    img = mark_faces(plt.imread(img_path), bndboxes, labels)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    plt.axis("off")
    ax.legend(title=image_name)
    ax.imshow(img)
    fig.savefig("test.png")
