import torch
from os import listdir

IMGS_PATH = './data/images/'
XMLS_PATH = './data/annotations/'
FILENAMES = listdir(IMGS_PATH)

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
