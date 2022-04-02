import xmltodict
import numpy as np
from matplotlib import pyplot as plt 
from os import listdir

METADATA_PATH = './data/annotations/'

# Get a paths for all annotation files
def get_metadata_paths():
  metadata_paths = []
  for filename in listdir(METADATA_PATH):
    metadata_paths.append(METADATA_PATH + filename)
  return metadata_paths

# Loop all annotation files and save mask data to list
data = []
for path in get_metadata_paths():
  with open(path) as file:
    # Parse to dict and select the 'object' element(s) 
    metadata = xmltodict.parse(file.read())
    obj = metadata['annotation']['object']

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

plt.savefig('images/mask_data.png')