import xmltodict
import cv2

from os import path

from globals import XMLS_PATH

classes = ["", "with_mask", "without_mask", "mask_weared_incorrect"]
box_colors = [(), (0, 255, 0), (0, 0, 255), (255, 0, 0)]


def get_annotation(filename, width=0, height=0, width_resized=0, height_resized=0):
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


def mark_faces(img, bboxes, labels):
    """
    Mark faces by drawing a box around the face.
    The box color is set depending on label value:
        with_mask = Green
        without_mask = Red
        mask_weared_incorrect = Blue
    """
    # Running cv2.cvtColor() twice seems inefficient, but duplicated boxes and wierd colors if not used.
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for bbox, label in zip(bboxes, labels):
        cv2.rectangle(
            img,
            (int(bbox[0]), int(bbox[1])),
            (int(bbox[2]), int(bbox[3])),
            color=box_colors[label],
            thickness=1,
        )
        # Todo: Print how sure prediction in %
        cv2.putText(
            img,
            classes[label],
            (int(bbox[0]), int(bbox[1] - 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            color=box_colors[label],
        )
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
