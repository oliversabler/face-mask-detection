import random
from os import path
from PIL import Image
from utils import get_annotation, plot_image
from globals import FILENAMES, IMGS_PATH

def _visualize_image(img_name):
    img_path = path.join(IMGS_PATH, img_name)

    bboxes, labels = get_annotation(img_name)

    img = Image.open(img_path).convert('RGB')

    plot_image(img, img_name, bboxes, labels)


def visualize_random_image():
    img_name = FILENAMES[random.randint(0, len(FILENAMES))]
    _visualize_image(img_name)


def visualize_image_by_index(index=0):
    img_name = FILENAMES[index]
    _visualize_image(img_name)
