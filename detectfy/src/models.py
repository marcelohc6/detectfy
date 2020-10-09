from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torch_tranforms import base_transform


MODELS = {
    'faster_rcnn': {'model': FastRCNNPredictor, 'transforms': base_transform},
}


def get_model_config(arch):
    assert arch in MODELS.keys(), 'Model architecture not found'
    return MODELS[arch]


class Model:
    def __init__(self):
        pass
    