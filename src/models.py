#region PRIVATE

#endregion
#region API

import warnings
from utils import *

def load_yolo_model(path: str = 'yolov8n.pt'):
    from ultralytics import YOLO # type: ignore
    
    model = YOLO(path)
    return model

def load_detr_model(device):
    from transformers import DetrForObjectDetection
    
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        
        model = DetrForObjectDetection.from_pretrained(
            "facebook/detr-resnet-50",
            num_labels=1, 
            ignore_mismatched_sizes=True 
        )

    return model.to(device)

def load_custom_model():
    return None

def load_rcnn_model(device):
    import torchvision
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features # type: ignore
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)

    return model.to(device)

#endregion