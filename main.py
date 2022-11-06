"""
Entry point of training

Folder dependencies:
    * /data/
    * /data/images/
    * /data/annotations/
    * /models/
"""
from os import listdir
from engine import train
from prediction import predict_random_image
from visualizers import visualize_random_image

if __name__ == '__main__':
    IMGS_PATH = './data/images/'
    XMLS_PATH = './data/annotations/'
    filenames = listdir(IMGS_PATH)

    train(filenames, IMGS_PATH, XMLS_PATH)
    # visualize_random_image(filenames, IMGS_PATH, XMLS_PATH)
    # predict_random_image(
    #     filenames,
    #     IMGS_PATH,
    #     XMLS_PATH,
    #     model_path="./models/model_2022-04-15 11 34 02.128306.pth",
    #     num_preds=1
    # )
