from xml.sax.saxutils import XMLFilterBase
import xmltodict
from matplotlib import pyplot as plt
from os import listdir

ANNOTATIONS_PATH = './data/annotations/'

# Populate annotation file path to list
ann_file_path = []
for filename in listdir(ANNOTATIONS_PATH):
  ann_file_path.append(ANNOTATIONS_PATH + filename)

# Loop all annotation files and save mask data to list
mask_data = []
for file in ann_file_path:
  with open(file) as f:
    # Parse to dict and select the 'object' element(s) 
    ann = xmltodict.parse(f.read())
    obj = ann['annotation']['object']

    # If the object is a list loop through it, else select name
    if type(obj) == list:
      for o in obj:
        mask_data.append(o['name'])
    else:
      mask_data.append(obj['name'])

# Todo: plot mask_data
print(mask_data.count('with_mask'))
print(mask_data.count('without_mask'))
print(mask_data.count('mask_weared_incorrect'))