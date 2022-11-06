"""
Utility functions
"""
import os
import xmltodict
from PIL import ImageDraw

from matplotlib import pyplot as plt

classes = ['', 'with_mask', 'without_mask', 'mask_weared_incorrect']
box_colors = ['', 'green', 'red', 'blue']

def get_annotation(filename, xmls_path, width=0, height=0, width_resized=1, height_resized=1):
    """
    Get annotation data in xml file for image based on filename.
        bboxes: face coordinates
        labels: mask wearing status
    """
    bboxes = []
    labels = []
    xml_path = os.path.join(xmls_path, filename[:-3] + 'xml')

    with open(xml_path, encoding='utf-8') as file:
        xml = xmltodict.parse(file.read())
        root = xml['annotation']

        obj = root['object']
        if not isinstance(obj, list):
            obj = [obj]

        for obj in obj:
            xmin, ymin, xmax, ymax = list(map(int, obj['bndbox'].values()))
            bboxes.append([xmin, ymin, xmax, ymax])

            label = obj['name']
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

def mark_faces(img, bboxes, labels):
    """
    Marks faces by drawing a box around the face
    """
    for bbox, label in zip(bboxes, labels):
        xmin, ymin, xmax, ymax = bbox
        shape = [
            (xmin, ymin),
            (xmax, ymax),
        ]
        draw = ImageDraw.Draw(img)
        draw.rectangle(shape, outline=box_colors[label])

    return img

def plot_image(img, img_name, bboxes, labels):
    """
    Mark faces by drawing a box around the face.
    The box color is set depending on label value:
        with_mask = Green
        without_mask = Red
        mask_weared_incorrect = Blue
    """
    _, ax = plt.subplots(1, 1)

    img = mark_faces(img, bboxes, labels)

    ax.legend(title=img_name)

    plt.axis('off')
    plt.imshow(img)
    plt.show()
