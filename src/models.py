#region PRIVATE

#endregion
#region API

from utils import *

def load_yolo_model(path: str = 'yolov8n.pt'):
    from ultralytics import YOLO # type: ignore
    
    model = YOLO(path)
    return model

def load_detr_model(dataloader=None, device=None):
    from transformers import DetrForObjectDetection, DetrFeatureExtractor
    
    feature_extractor = DetrFeatureExtractor.from_pretrained("facebook/detr-resnet-50")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
    model = model.to(device) # type: ignore

    return model, feature_extractor

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