import xmltodict
import cv2

from os import path

from globals import XMLS_PATH

"""
Object labels as constants
"""
WITH_MASK = "with_mask"
WITHOUT_MASK = "without_mask"
MASK_WEARED_INCORRECT = "mask_weared_incorrect"


def get_label_id(label):
    """
    Convert label to an integer:
        WITH_MASK = 0
        WITHOUT_MASK = 1
        MASK_WEARED_INCORRECT = 2
    """
    if label == WITH_MASK:
        label_id = 2
    elif label == WITHOUT_MASK:
        label_id = 1
    elif label == MASK_WEARED_INCORRECT:
        label_id = 0
    return label_id


def get_annotation(filename):
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
            labels.append(get_label_id(label))

        return bboxes, labels


def get_box_color(label):
    """
    Set box color depending on label:
        0 = Green
        1 = Red
        2 = Blue
    """
    if label == 2:
        color = (0, 255, 0)
    elif label == 1:
        color = (0, 0, 255)
    else:
        color = (255, 0, 0)
    return color


def mark_faces(img, bboxes, labels):
    """
    Mark faces by drawing a box around the face.
    The box color is set depending on label value:
        WITH_MASK = Green
        WITHOUT_MASK = Red
        MASK_WEARED_INCORRECT = Blue
    """
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    for bbox, label in zip(bboxes, labels):
        cv2.rectangle(
            img,
            (int(bbox[0]), int(bbox[1])),
            (int(bbox[2]), int(bbox[3])),
            color=get_box_color(label),
            thickness=1,
        )
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
