import os
import sys
import matplotlib.pyplot as plt
import loaders
import torch
import inference
import models
import utils


BASE_PATH = os.path.abspath('')
print(BASE_PATH)

INPUT_PATH = os.path.join(os.path.abspath('').split('src')[0], 'input')
BATCH_SIZE = 2

img_mask_folder = 'PennFudanPed'
img_mask_path = os.path.join(INPUT_PATH, img_mask_folder)


dataset = loaders.BaseDataset(img_mask_path, 'PNGImages', 'PedMasks')

test1 = dataset[0]
test2 = dataset[1]

print(test1[0].size())
# print(test1)

baseloader = loaders.BaseDataLoader(dataset, batch_size=BATCH_SIZE)

# print(next(iter(baseloader)))


