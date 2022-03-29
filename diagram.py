import xmltodict
import numpy as np
from matplotlib import pyplot as plt 
from os import listdir

ANNOTATIONS_PATH = './data/annotations/'

# Populate annotation file path to list
ann_file_path = []
for filename in listdir(ANNOTATIONS_PATH):
  ann_file_path.append(ANNOTATIONS_PATH + filename)

# Loop all annotation files and save mask data to list
data = []
for file in ann_file_path:
  with open(file) as f:
    # Parse to dict and select the 'object' element(s) 
    ann = xmltodict.parse(f.read())
    obj = ann['annotation']['object']

    # If the object is a list loop through it, else select name
    if type(obj) == list:
      for o in obj:
        data.append(o['name'])
    else:
      data.append(obj['name'])

# Plot mask data
x = ['With Mask', 'Without Mask', 'Mask Weared Incorrect']
y = [data.count('with_mask'), data.count('without_mask'), data.count('mask_weared_incorrect')]

fig, ax = plt.subplots()
ax.bar(x, y)

plt.savefig('mask_data.png')