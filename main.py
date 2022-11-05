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

if __name__ == '__main__':
    imgs_path = './data/images/'
    xmls_path = './data/annotations/'
    filenames = listdir(imgs_path)
    train(filenames, imgs_path, xmls_path)
