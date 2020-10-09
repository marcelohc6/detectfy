from loaders import BaseDataset, BaseDataLoader
from models import get_model_config
import os



class Model():






if __name__ == '__main__':
    arch = 'faster_rcnn'
    model_cfg = get_model_config(arch)
    model = model_cfg['model']
    dataset = BaseDataset(img_mask_path, 'PNGImages', 'PedMasks', transform=model_cfg['transforms'])
    baseloader = BaseDataLoader(dataset, batch_size=BATCH_SIZE)




