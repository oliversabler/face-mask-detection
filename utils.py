import xmltodict
import cv2
from os import path

WITH_MASK = "with_mask"
WITHOUT_MASK = "without_mask"
MASK_WEARED_INCORRECT = "mask_weared_incorrect"


def get_label_id(label):
    if label == WITH_MASK:
        label_id = 2
    elif label == WITHOUT_MASK:
        label_id = 1
    elif label == MASK_WEARED_INCORRECT:
        label_id = 0
    return label_id


def get_box_color(label):
    if label == 2:
        color = (0, 255, 0)
    elif label == 1:
        color = (0, 0, 255)
    else:
        color = (255, 0, 0)
    return color


def get_annotation(filename, xmls_path):
    bndboxes = []
    labels = []

    xml_path = path.join(xmls_path, filename[:-3] + "xml")

    with open(xml_path) as file:
        xml = xmltodict.parse(file.read())
        root = xml["annotation"]

        obj = root["object"]
        if type(obj) != list:
            obj = [obj]

        for obj in obj:
            xmin, ymin, xmax, ymax = list(map(int, obj["bndbox"].values()))
            bndboxes.append([xmin, ymin, xmax, ymax])

            label = obj["name"]
            labels.append(get_label_id(label))

        return bndboxes, labels


def mark_faces(img, bndboxes, labels):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    for bndbox, label in zip(bndboxes, labels):
        cv2.rectangle(
            img,
            (int(bndbox[0]), int(bndbox[1])),
            (int(bndbox[2]), int(bndbox[3])),
            color=get_box_color(label),
            thickness=1,
        )
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
