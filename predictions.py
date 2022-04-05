import random

from PIL import Image
from os import path

import torch
import torchvision
import torchvision.transforms as transforms

from matplotlib import pyplot as plt

from models import load_resnet50_model_state
from utils import get_annotation, mark_faces


def predict_img(img, nm_thrs=0.3, score_thrs=0.8, device="cpu"):
    test_img = transforms.ToTensor()(img)

    model = load_resnet50_model_state("./models/model_2022-04-05 05:05:46.048959.pth")
    model.eval()

    with torch.no_grad():
        predictions = model(test_img.unsqueeze(0).to(device))

    test_img = test_img.permute(1, 2, 0).numpy()
    keep_boxes = torchvision.ops.nms(
        predictions[0]["boxes"].cpu(), predictions[0]["scores"].cpu(), nm_thrs
    )

    score_filter = predictions[0]["scores"].cpu().numpy()[keep_boxes] > score_thrs

    test_boxes = predictions[0]["boxes"].cpu().numpy()[keep_boxes][score_filter]
    test_labels = predictions[0]["labels"].cpu().numpy()[keep_boxes][score_filter]

    return test_img, test_boxes, test_labels


def predict_random_image(filenames, imgs_path, xmls_path):
    random_image_name = filenames[random.randint(0, len(filenames))]
    test_img = Image.open(path.join(imgs_path, random_image_name)).convert("RGB")

    # Prediction
    test_img, test_boxes, test_labels = predict_img(test_img)
    test_output = mark_faces(test_img, test_boxes, test_labels)

    # Solution
    bndbox, labels = get_annotation(random_image_name, xmls_path)
    gt_output = mark_faces(test_img, bndbox, labels)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    ax1.imshow(test_output)
    ax1.set_xlabel("Prediction")
    ax2.imshow(gt_output)
    ax2.set_xlabel("Truth")
    plt.savefig("prediction.png")
