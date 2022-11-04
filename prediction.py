"""
Handles prediction
"""
import random
from os import path
from PIL import Image

import torch
import torchvision
from torchvision import transforms

from matplotlib import pyplot as plt
from globals import FILENAMES, IMGS_PATH, DEVICE
from model import load_resnet50_model_state
from utils import get_annotation, mark_faces

def _predict_img(model_path, img, nm_thrs=0.3, score_thrs=0.8):
    img = transforms.ToTensor()(img)

    model = load_resnet50_model_state(model_path)
    model.eval()

    with torch.no_grad():
        predictions = model(img.unsqueeze(0).to(DEVICE))

    keep_boxes = torchvision.ops.nms(
        predictions[0]['boxes'].cpu(), predictions[0]['scores'].cpu(), nm_thrs
    )

    score_filter = predictions[0]['scores'].cpu().numpy()[keep_boxes] > score_thrs

    boxes = predictions[0]['boxes'].cpu().numpy()[keep_boxes][score_filter]
    labels = predictions[0]['labels'].cpu().numpy()[keep_boxes][score_filter]

    print(f'Prediction labels: {labels}')

    img = transforms.ToPILImage()(img)

    return img, boxes, labels

def predict_random_image(model_path, num_preds=1):
    for i in range(num_preds):
        img_name = FILENAMES[random.randint(0, len(FILENAMES))]
        img_path = path.join(IMGS_PATH, img_name)

        img = Image.open(img_path).convert('RGB')

        # Prediction
        p_img, p_boxes, p_labels = _predict_img(model_path, img)
        p_output = mark_faces(p_img, p_boxes, p_labels)

        # Solution
        t_boxes, t_labels = get_annotation(img_name)
        t_output = mark_faces(img, t_boxes, t_labels)

        _, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        ax1.imshow(p_output)
        ax1.set_xlabel('Prediction')
        ax2.imshow(t_output)
        ax2.set_xlabel('Truth')
