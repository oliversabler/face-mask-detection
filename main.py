from os import listdir
from engine import train

if __name__ == '__main__':
    # Todo: Create folder structure
    imgs_path = './data/images/'
    xmls_path = './data/annotations/'
    filenames = listdir(imgs_path)
    train(filenames)
