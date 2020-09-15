
import os
import numpy as np
import torch
from PIL import Image
import cv2

from config import IMG_ENGINE


def _pil_load_image_mask(img_path, mask_path):
    img = Image.open(img_path).convert("RGB")
    # note that we haven't converted the mask to RGB,
    # because each color corresponds to a different instance
    # with 0 being background
    mask = Image.open(mask_path)
    # convert the PIL Image into a numpy array
    return img, np.array(mask)


ultils_funcs = {'load_img_mask':
                    {'engine': 
                        {'pil': _pil_load_image_mask,
                         'cv2': None,
                        }
                    }
                }


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, img_root, img_folder, mask_folder, transforms):
        self.img_root = img_root
        self.img_folder = img_folder
        self.mask_folder = mask_folder
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(img_root, img_folder))))
        self.masks = list(sorted(os.listdir(os.path.join(img_root, mask_folder))))

    def _get_img_mask(self, img_path, mask_path):
        return ultils_funcs['load_img_mask']['engine'][IMG_ENGINE](img_path, mask_path)

    def __getitem__(self, idx):
        # load images ad masks
        img_path = os.path.join(self.img_root, self.img_folder, self.imgs[idx])
        mask_path = os.path.join(self.mask_root, self.mask_folder, self.masks[idx])

        # Load images and masks
        img, mask = self._get_img_mask(img_path, mask_path)

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

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)
