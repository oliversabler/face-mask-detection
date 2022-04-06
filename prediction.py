import random
import cv2

from os import path

from matplotlib import pyplot as plt
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms

from globals import FILENAMES, IMGS_PATH, DEVICE
from model import load_resnet50_model_state
from utils import get_annotation, mark_faces


def predict_img(img, nm_thrs=0.3, score_thrs=0.8):
    img = transforms.ToTensor()(img)

    model = load_resnet50_model_state("./models/model_2022-04-05 05:05:46.048959.pth")
    model.eval()

    with torch.no_grad():
        predictions = model(img.unsqueeze(0).to(DEVICE))

    img = img.permute(1, 2, 0).numpy()

    keep_boxes = torchvision.ops.nms(
        predictions[0]["boxes"].cpu(), predictions[0]["scores"].cpu(), nm_thrs
    )

    score_filter = predictions[0]["scores"].cpu().numpy()[keep_boxes] > score_thrs

    boxes = predictions[0]["boxes"].cpu().numpy()[keep_boxes][score_filter]
    labels = predictions[0]["labels"].cpu().numpy()[keep_boxes][score_filter]

    return img, boxes, labels


def predict_random_image():
    # Read image name by index and construct path
    img_name = FILENAMES[random.randint(0, len(FILENAMES))]
    img_path = path.join(IMGS_PATH, img_name)

    # Open image
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = cv2.resize(img, (480, 480), cv2.INTER_AREA)
    # img /= 255.0

    # Prediction
    img, boxes, labels = predict_img(img)
    p_output = mark_faces(img, boxes, labels)

    # Solution
    bboxes, labels = get_annotation(img_name)
    t_output = mark_faces(img, bboxes, labels)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    ax1.imshow(p_output)
    ax1.set_xlabel("Prediction")
    ax2.imshow(t_output)
    ax2.set_xlabel("Truth")
    fig.savefig("prediction.png")
