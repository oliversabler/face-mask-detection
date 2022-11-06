import random
from os import path
from PIL import Image
from utils import get_annotation, plot_image

def _visualize_image(imgs_path, xmls_path, img_name):
    img_path = path.join(imgs_path, img_name)

    bboxes, labels = get_annotation(img_name, xmls_path)

    img = Image.open(img_path).convert('RGB')

    plot_image(img, img_name, bboxes, labels)


def visualize_random_image(filenames, imgs_path, xmls_path):
    img_name = filenames[random.randint(0, len(filenames))]
    _visualize_image(imgs_path, xmls_path, img_name)


def visualize_image_by_index(filenames, imgs_path, xmls_path, index=0):
    img_name = filenames[index]
    _visualize_image(imgs_path, xmls_path, img_name)
