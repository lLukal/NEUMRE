import sys

from tqdm import tqdm

from data import *
from models import *
from utils import *
from torch.utils.data import DataLoader
import torch
import cv2
import numpy as np
import os
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import warnings
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from models import CustomPedestrianDetector

#region HELPERS

# helper to convert box formats
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

class RCNNPredictor:
    def __init__(self, weight_path, num_classes=2):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features # type: ignore
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        
        self.model.load_state_dict(torch.load(weight_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)


    def predict(self, image_path, confidence_threshold=0.5):
        img = cv2.imread(image_path)
        if img is None:
            return None

        orig_h, orig_w = img.shape[:2]
        
        image_w = 800
        image_h = int(round(orig_h / orig_w * image_w))

        img_resized = cv2.resize(img, (image_w, image_h))
        
        img_prep = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB).astype(np.float32)
        img_prep = cv2.resize(img_prep, (image_w, image_h))

        img_prep /= 255.0        
        img_prep = (img_prep - self.mean) / self.std
        
        img_tensor = torch.as_tensor(img_prep).permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            predictions = self.model(img_tensor)[0]

        for i, score in enumerate(predictions['scores']):
            if score > confidence_threshold:
                box = predictions['boxes'][i].cpu().numpy()

                cv2.rectangle(img_resized, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
                cv2.putText(img_resized, f"{score:.2f}", (int(box[0]), int(box[1]-5)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return img_resized

#endregion
#region API

def evaluate_yolo(model, dataset_type):
    metrics = model.val(data=f'../data/yolo/{dataset_type.value}/dataset.yaml')
    print(metrics.box.map)
    print(metrics.box.map50)
    print(metrics.box.precision)
    print(metrics.box.recall)

def evaluate_detr(model, dataloader, device, output_dir="./runs/detr", epoch_num=None):
    warnings.filterwarnings("ignore", message=".*meta parameter.*") 

    model.eval()
    os.makedirs(f"{output_dir}/viz", exist_ok=True)
    
    # initialize metric
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
    
    # visualization limits
    viz_count = 0
    max_viz = 5 # reduced for speed during training loops
    
    # ImageNet stats for un-normalization (visuals)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(tqdm(dataloader, desc="Validating", leave=False)):
            images = images.to(device)
            
            # forward pass
            outputs = model(pixel_values=images)
            
            pred_logits = outputs.logits
            pred_boxes = outputs.pred_boxes

            metric_preds = []
            metric_targets = []

            batch_size = images.shape[0]
            
            for i in range(batch_size):
                h, w = images[i].shape[1], images[i].shape[2]
                
                tgt_boxes_norm = targets[i]['boxes'].to(device)
                tgt_boxes_abs = box_cxcywh_to_xyxy(tgt_boxes_norm)
                tgt_boxes_abs = tgt_boxes_abs * torch.tensor([w, h, w, h], device=device)
                
                metric_targets.append({
                    "boxes": tgt_boxes_abs,
                    "labels": targets[i]['class_labels'].to(device)
                })

                probas = pred_logits[i].softmax(-1)
                scores, labels = probas[:, :-1].max(-1)
                
                p_boxes_norm = pred_boxes[i]
                p_boxes_abs = box_cxcywh_to_xyxy(p_boxes_norm)
                p_boxes_abs = p_boxes_abs * torch.tensor([w, h, w, h], device=device)

                metric_preds.append({
                    "boxes": p_boxes_abs,
                    "scores": scores,
                    "labels": labels
                })

                # only visualize a few samples from the first batch
                if viz_count < max_viz and epoch_num is not None:
                    keep = scores > 0.5
                    if keep.sum() > 0:
                        viz_boxes = p_boxes_abs[keep].cpu().numpy()
                        viz_labels = labels[keep].cpu().numpy()
                        viz_scores = scores[keep].cpu().numpy()

                        img_viz = images[i] * std + mean
                        img_viz = torch.clamp(img_viz, 0, 1)
                        
                        img_np = img_viz.permute(1, 2, 0).cpu().numpy()
                        img_np = (img_np * 255).astype(np.uint8)
                        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

                        for box, label, score in zip(viz_boxes, viz_labels, viz_scores):
                            x1, y1, x2, y2 = map(int, box)
                            cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(img_np, f"{score:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        viz_path = f"{output_dir}/viz/epoch_{epoch_num}_batch_{batch_idx}_img_{i}.jpg"
                        cv2.imwrite(viz_path, img_np)
                        viz_count += 1

            # update metric
            metric.update(metric_preds, metric_targets)

    # compute results
    result = metric.compute()
    return result

def evaluate_rcnn(model, val_loader, device):
    metric = MeanAveragePrecision()
    val_loss = 0

    with torch.no_grad():
        for images, targets in val_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            model.train()
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            val_loss += losses.item() # type: ignore

            model.eval()
            outputs = model(images)

            res = [{k: v.to(device) for k, v in t.items()} for t in outputs]
            gt = [{k: v.to(device) for k, v in t.items()} for t in targets]
            metric.update(res, gt)

    map_results = metric.compute()
    return (val_loss / len(val_loader)), map_results


def evaluate_custom(model, val_loader, device):
    """Evaluate custom model and compute mAP metrics."""
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
    val_loss = 0
    num_batches = 0

    for images, targets in tqdm(val_loader, desc="Evaluating Custom Model"):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Compute validation loss
        model.train()
        with torch.no_grad():
            loss_dict = model(images, targets)
            loss = loss_dict['loss']
            val_loss += loss.item()
            num_batches += 1

        # Get predictions for mAP
        model.eval()
        with torch.no_grad():
            outputs = model(images)

        # Format predictions and targets for torchmetrics
        preds = []
        gts = []
        for out, tgt in zip(outputs, targets):
            preds.append({
                'boxes': out['boxes'].to(device),
                'scores': out['scores'].to(device),
                'labels': out['labels'].to(device),
            })
            gts.append({
                'boxes': tgt['boxes'].to(device),
                'labels': tgt['labels'].to(device),
            })
        
        metric.update(preds, gts)

    map_results = metric.compute()
    avg_val_loss = val_loss / max(num_batches, 1)
    return avg_val_loss, map_results


class CustomPredictor:
    """Predictor class for custom model, compatible with explore.py visualization."""
    def __init__(self, weight_path, num_classes=2, input_size=640):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.input_size = input_size
        
        self.model = CustomPedestrianDetector(
            num_classes=num_classes,
            num_anchors=9,
            input_size=input_size
        )
        
        self.model.load_state_dict(torch.load(weight_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def predict(self, image_path, confidence_threshold=0.9):
        """Run inference on a single image and return annotated image."""
        img = cv2.imread(image_path)
        if img is None:
            return None

        orig_h, orig_w = img.shape[:2]
        
        # Resize to input size
        img_resized = cv2.resize(img, (self.input_size, self.input_size))
        
        # Prepare tensor
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(img_tensor)[0]

        boxes = outputs['boxes'].cpu().numpy()
        scores = outputs['scores'].cpu().numpy()
        labels = outputs['labels'].cpu().numpy()

        for box, score, label in zip(boxes, scores, labels):
            if score > confidence_threshold:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(img_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img_resized, f"{score:.2f}", (x1, y1-5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return img_resized

    def predict_tensor(self, image):
        """Run inference on a tensor/numpy image and return raw outputs."""
        if isinstance(image, np.ndarray):
            img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            img_resized = cv2.resize(img_rgb, (self.input_size, self.input_size))
            img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).unsqueeze(0).to(self.device)
        else:
            img_tensor = image.to(self.device)

        with torch.no_grad():
            outputs = self.model(img_tensor)[0]
        
        return outputs

#endregion
#region MAIN

def run_pipeline(dataset_type: DatasetType, model_type: ModelType, weights_path: str):
    print('Starting training pipeline...')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"

    try:
        # load and prepare dataset
        print(f'Preparing dataset "{dataset_type.value}"...')
        if model_type != ModelType.YOLO:
            try:
                dataloader = load_dataset(dataset_type, model_type, 'test')
            except:
                dataloader = load_dataset(dataset_type, model_type, 'val')

        # load and evaluate model
        print(f'\tPreparing model "{model_type.value}"...')
        if model_type == ModelType.YOLO:
            model = load_yolo_model(weights_path)
            evaluate_yolo(model, dataset_type)


        elif model_type == ModelType.DETR:
            model = load_detr_model(dataloader)
            evaluate_detr(model, dataloader, device)


        elif model_type == ModelType.RCNN:
            model = load_rcnn_model(device)
            evaluate_rcnn(model, dataloader, device)


        elif model_type == ModelType.CUSTOM:
            model = load_custom_model(device=device, num_classes=2)
            model.load_state_dict(torch.load(weights_path, map_location=device))
            model.to(device)
            val_loss, map_results = evaluate_custom(model, dataloader, device)
            print(f"Validation Loss: {val_loss:.4f}")
            print(f"mAP: {map_results['map'].item():.4f}")
            print(f"mAP@50: {map_results['map_50'].item():.4f}")
            print(f"mAP@75: {map_results['map_75'].item():.4f}")

    except Exception as e:
        print(e)

if __name__ == '__main__':
    import torch
    torch.cuda.empty_cache()
    import gc
    gc.collect()

    # parse system parameters
    args = sys.argv
    if len(args) < 4:
        print('Usage: python train.py <dataset> <model> <weights_path>')
        print(f'<dataset> - dataset name, options:\n\t\t' + '\n\t\t'.join([d.value for d in DatasetType]))
        print(f'<model> - model type, options:\n\t\t' + '\n\t\t'.join([m.value for m in ModelType]))
        print(f'<weights_path> - filepath to weights')
        exit(1)

    # run
    run_pipeline(DatasetType(args[1]), ModelType(args[2]), args[3])
