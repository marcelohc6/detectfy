
import os
import numpy as np
import torch
from PIL import Image
import cv2
from torch_tranforms import base_transform

import config

def _pil_load_image_mask(img_path, mask_path):
    img = Image.open(img_path).convert("RGB")
    # note that we haven't converted the mask to RGB,
    # because each color corresponds to a different instance
    # with 0 being background
    mask = Image.open(mask_path)
    # convert the PIL Image into a numpy array
    return img, np.array(mask)


def _cv2_load_image_mask(img_path, mask_path):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    return img


ultils_funcs = {'load_img_mask':
                    {'engine': 
                        {'pil': _pil_load_image_mask,
                         'cv2': _cv2_load_image_mask,
                        }
                    }
                }


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, img_root, img_folder, mask_folder, transform=None, aumentation=False):
        self.img_root = img_root
        self.img_folder = img_folder
        self.mask_folder = mask_folder
        self.transform = transform if transform is not None else base_transform
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(img_root, img_folder))))
        self.masks = list(sorted(os.listdir(os.path.join(img_root, mask_folder))))

    def _get_img_mask(self, img_path, mask_path):
        return ultils_funcs['load_img_mask']['engine'][config.IMG_ENGINE](img_path, mask_path)

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.img_root, self.img_folder, self.imgs[idx])
        mask_path = os.path.join(self.img_root, self.mask_folder, self.masks[idx])

        # Load images and masks
        img, mask = self._get_img_mask(img_path, mask_path)

        # Image to np.array
        #img = np.array(img)

        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.imgs)



def custom_collate_data(batch):
    img, target = zip(*batch)
    return img, target


class BaseDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, **kwargs):
        super().__init__(dataset, collate_fn=custom_collate_data, **kwargs)
    