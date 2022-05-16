import xmltodict

from os import path
from globals import XMLS_PATH

from matplotlib import pyplot as plt
import matplotlib.patches as patches


class Logger:
    def __init__(self, name, epoch, iteration=0):
        self.name = name
        self.epoch = epoch
        self.iteration = iteration
        self.metrics = {}

    def __str__(self):
        msg = "{} | Epoch [{}] Iteration [{}] - ".format(
            self.name, self.epoch, self.iteration
        )

        for k, v in sorted(self.metrics.items()):
            if type(v) is float:
                msg += "{}: {:.4}, ".format(k, v)
            else:
                msg += "{}: {}, ".format(k, v)

        return msg

    def increment(self):
        self.iteration += 1

    def update(self, **kwargs):
        for k, v in kwargs.items():
            self.metrics[k] = v


classes = ["", "with_mask", "without_mask", "mask_weared_incorrect"]
box_colors = ["", "g", "r", "b"]


def get_annotation(filename, width=0, height=0, width_resized=1, height_resized=1):
    """
    Get annotation data in xml file for image based on filename.
        bboxes: face coordinates
        labels: mask wearing status
    """
    bboxes = []
    labels = []

    xml_path = path.join(XMLS_PATH, filename[:-3] + "xml")

    with open(xml_path) as file:
        xml = xmltodict.parse(file.read())
        root = xml["annotation"]

        obj = root["object"]
        if type(obj) != list:
            obj = [obj]

        for obj in obj:
            xmin, ymin, xmax, ymax = list(map(int, obj["bndbox"].values()))
            bboxes.append([xmin, ymin, xmax, ymax])

            label = obj["name"]
            labels.append(classes.index(label))

        if width != 0 and height != 0:
            boxes_corr = []
            for box in bboxes:
                xmin_corr = (box[0] / width) * width_resized
                xmax_corr = (box[2] / width) * width_resized
                ymin_corr = (box[1] / height) * height_resized
                ymax_corr = (box[3] / height) * height_resized
                boxes_corr.append([xmin_corr, ymin_corr, xmax_corr, ymax_corr])

            return boxes_corr, labels

        return bboxes, labels


def plot_image(img, img_name, boxes, labels):
    """
    Mark faces by drawing a box around the face.
    The box color is set depending on label value:
        with_mask = Green
        without_mask = Red
        mask_weared_incorrect = Blue
    """
    fig, ax = plt.subplots(1, 1)

    ax.imshow(img.permute(1, 2, 0))

    for bbox, label in zip(boxes, labels):
        xmin, ymin, xmax, ymax = bbox
        rect = patches.Rectangle(
            (xmin, ymin),
            (xmax - xmin),
            (ymax - ymin),
            linewidth=1,
            edgecolor=box_colors[label],
            facecolor="none",
        )
        ax.add_patch(rect)

    plt.axis("off")
    ax.legend(title=img_name)
    fig.savefig("./temp/visualization.png", bbox_inches="tight", pad_inches=0)

    return ax
