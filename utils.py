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


def _get_label_id(label):
    """
    Convert label to an integer:
        WITH_MASK = 1
        WITHOUT_MASK = 2
        MASK_WEARED_INCORRECT = 3
    """
    if label == WITH_MASK:
        label_id = 1
    elif label == WITHOUT_MASK:
        label_id = 2
    elif label == MASK_WEARED_INCORRECT:
        label_id = 3
    return label_id


def get_annotation(filename, width=0, height=0, width_resized=0, height_resized=0):
    """
    Get annotation data in xml file for image based on filename.
        bboxes: face coordinates
        labels: mask wearing status
    """
    boxes = []
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
            boxes.append([xmin, ymin, xmax, ymax])

            label = obj["name"]
            labels.append(_get_label_id(label))

        if width != 0 and height != 0:
            boxes_corr = []
            for box in boxes:
                xmin_corr = (box[0] / width) * width_resized
                xmax_corr = (box[2] / width) * width_resized
                ymin_corr = (box[1] / height) * height_resized
                ymax_corr = (box[3] / height) * height_resized
                boxes_corr.append([xmin_corr, ymin_corr, xmax_corr, ymax_corr])

            return boxes_corr, labels

        return boxes, labels


def _get_box_color(label):
    """
    Set box color depending on label (cv2 uses BGR instead of RGB):
        1 = Green
        2 = Red
        3 = Blue
    """
    if label == 1:
        color = (0, 255, 0)
    elif label == 2:
        color = (0, 0, 255)
    elif label == 3:
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
    # Running cv2.cvtColor() twice seems inefficient, but duplicated boxes and wierd colors if not used.
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for bbox, label in zip(bboxes, labels):
        cv2.rectangle(
            img,
            (int(bbox[0]), int(bbox[1])),
            (int(bbox[2]), int(bbox[3])),
            color=_get_box_color(label),
            thickness=1,
        )
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
