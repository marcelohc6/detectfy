from loaders import BaseDataset
import os


INPUT_PATH = os.path.join(os.path.abspath('').split('src')[0], 'input')

img_mask_folder = 'PennFudanPed'
img_mask_path = os.path.join(INPUT_PATH, img_mask_folder)

dataset = BaseDataset(img_mask_path, 'PNGImages', 'PedMasks')

test = dataset[0]


print(test[0])
print(test[1])



