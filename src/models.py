import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.ops import nms, box_iou
import warnings
from utils import *
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

#endregion
#region PRIVATE

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

class CustomPedestrianDetector(nn.Module):
    """
    Faster R-CNN-style model for pedestrian detection.
    
    Architecture:
        - Backbone: Lightweight CNN feature extractor
        - RPN: Region Proposal Network for generating object proposals
        - ROI Head: ROI Align + classification and box regression heads
    
    Compatible with PyTorch DataLoader returning (images, targets)
    """
    def __init__(self, num_classes=2, num_anchors=9, input_size=1024):
        super(CustomPedestrianDetector, self).__init__()
        
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.input_size = input_size
        self.feature_channels = 128
        self.roi_output_size = 7
        
        self.anchor_ratios = [0.5, 1.0, 2.0]
        self.anchor_scales = [16, 32, 64]
        
        # Backbone
        self.backbone = self._make_backbone()
        
        # Region Proposal Network (RPN)
        self.rpn_conv = nn.Conv2d(self.feature_channels, 256, kernel_size=3, padding=1)
        self.rpn_cls = nn.Conv2d(256, num_anchors, kernel_size=1)  # objectness
        self.rpn_reg = nn.Conv2d(256, num_anchors * 4, kernel_size=1)  # box deltas
        
        # ROI Head
        roi_input_features = self.feature_channels * self.roi_output_size * self.roi_output_size
        self.roi_head = nn.Sequential(
            nn.Linear(roi_input_features, 128),
            nn.ReLU(inplace=True),
        )
        self.cls_head = nn.Linear(128, num_classes)
        self.box_head = nn.Linear(128, num_classes * 4)
        
        # Loss weights
        self.rpn_lambda = 1.0
        self.roi_lambda = 1.0
        
    def _make_backbone(self):
        return nn.Sequential(
            # Stage 1: input -> /2
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            # Stage 2: /2 -> /4
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # Stage 3: /4 -> /8
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # Stage 4: /8 -> /16
            nn.Conv2d(128, self.feature_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.feature_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, images, targets=None):
        if isinstance(images, list):
            images = self._prepare_images(images)
        
        batch_size = images.shape[0]
        device = images.device
        
        # Extract features with backbone
        features = self.backbone(images)
        feat_h, feat_w = features.shape[2], features.shape[3]
        
        # RPN forward
        rpn_features = F.relu(self.rpn_conv(features))
        rpn_cls_logits = self.rpn_cls(rpn_features)  # [B, num_anchors, H, W]
        rpn_box_deltas = self.rpn_reg(rpn_features)  # [B, num_anchors*4, H, W]
        
        # Generate anchors
        anchors = self._generate_anchors(feat_h, feat_w, device)
        
        if self.training and targets is not None:
            return self._compute_loss(
                features, rpn_cls_logits, rpn_box_deltas, 
                anchors, targets, feat_h, feat_w, device
            )
        else:
            return self._inference(
                features, rpn_cls_logits, rpn_box_deltas,
                anchors, feat_h, feat_w, device
            )
    
    def _prepare_images(self, images):
        processed = []
        for img in images:
            img = F.interpolate(
                img.unsqueeze(0), 
                size=(self.input_size, self.input_size), 
                mode='bilinear', 
                align_corners=False
            )
            processed.append(img)
        return torch.cat(processed, dim=0)
    
    def _generate_anchors(self, feat_h, feat_w, device):
        stride_h = self.input_size / feat_h
        stride_w = self.input_size / feat_w
        
        anchors = []
        for y in range(feat_h):
            for x in range(feat_w):
                cx = (x + 0.5) * stride_w
                cy = (y + 0.5) * stride_h
                for scale in self.anchor_scales:
                    for ratio in self.anchor_ratios:
                        w = scale * (ratio ** 0.5)
                        h = scale / (ratio ** 0.5)
                        anchors.append([cx - w/2, cy - h/2, cx + w/2, cy + h/2])
        
        return torch.tensor(anchors, device=device, dtype=torch.float32)
    
    def _apply_box_deltas(self, anchors, deltas):
        # anchors: [N, 4] (x1, y1, x2, y2)
        # deltas: [N, 4] (dx, dy, dw, dh)
        widths = anchors[:, 2] - anchors[:, 0]
        heights = anchors[:, 3] - anchors[:, 1]
        cx = anchors[:, 0] + 0.5 * widths
        cy = anchors[:, 1] + 0.5 * heights
        
        dx, dy, dw, dh = deltas[:, 0], deltas[:, 1], deltas[:, 2], deltas[:, 3]
        
        pred_cx = dx * widths + cx
        pred_cy = dy * heights + cy
        pred_w = torch.exp(dw.clamp(max=4.0)) * widths
        pred_h = torch.exp(dh.clamp(max=4.0)) * heights
        
        pred_x1 = pred_cx - 0.5 * pred_w
        pred_y1 = pred_cy - 0.5 * pred_h
        pred_x2 = pred_cx + 0.5 * pred_w
        pred_y2 = pred_cy + 0.5 * pred_h
        
        return torch.stack([pred_x1, pred_y1, pred_x2, pred_y2], dim=1)
    
    def _roi_align(self, features, boxes, output_size=7):
        from torchvision.ops import roi_align
        return roi_align(features, boxes, output_size, spatial_scale=1.0/16.0) # type: ignore
    
    def _compute_loss(self, features, rpn_cls, rpn_reg, anchors, targets, feat_h, feat_w, device):
        batch_size = features.shape[0]

        # Reshape RPN outputs
        rpn_cls = rpn_cls.permute(0, 2, 3, 1).contiguous().view(batch_size, -1)
        rpn_reg = rpn_reg.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 4)

        rpn_cls_losses = []
        rpn_reg_losses = []
        roi_cls_losses = []
        roi_reg_losses = []

        for b in range(batch_size):
            target = targets[b]
            gt_boxes = target['boxes']
            gt_labels = target['labels']
            
            if gt_boxes.shape[0] == 0:
                continue
            
            # Compute IoU between anchors and ground truth
            ious = box_iou(anchors, gt_boxes)
            max_iou, matched_gt_idx = ious.max(dim=1)
            
            # RPN labels: 1=positive, 0=negative, -1=ignore
            rpn_labels = torch.full((anchors.shape[0],), -1, device=device)
            rpn_labels[max_iou < 0.3] = 0
            rpn_labels[max_iou >= 0.7] = 1
            
            # Also assign best anchor for each GT as positive
            best_anchor_per_gt = ious.argmax(dim=0)
            rpn_labels[best_anchor_per_gt] = 1
            
            # RPN classification loss
            pos_mask = rpn_labels == 1
            neg_mask = rpn_labels == 0
            
            if pos_mask.any() or neg_mask.any():
                rpn_cls_targets = torch.zeros_like(rpn_labels, dtype=torch.float32)
                rpn_cls_targets[pos_mask] = 1.0
                valid_mask = pos_mask | neg_mask
                rpn_cls_loss = F.binary_cross_entropy_with_logits(
                    rpn_cls[b][valid_mask], rpn_cls_targets[valid_mask], reduction='mean'
                )
                rpn_cls_losses.append(rpn_cls_loss)
            
            # RPN regression loss (only for positive anchors)
            if pos_mask.any():
                pos_anchors = anchors[pos_mask]
                pos_gt_boxes = gt_boxes[matched_gt_idx[pos_mask]]

                # Compute regression targets
                target_deltas = self._compute_box_deltas(pos_anchors, pos_gt_boxes)
                pred_deltas = rpn_reg[b][pos_mask]

                rpn_reg_loss = F.smooth_l1_loss(pred_deltas, target_deltas, reduction='mean')
                rpn_reg_losses.append(rpn_reg_loss)
            
            # Generate proposals for ROI head
            with torch.no_grad():
                proposals = self._apply_box_deltas(anchors, rpn_reg[b])
                proposals = proposals.clamp(min=0, max=self.input_size)
                scores = torch.sigmoid(rpn_cls[b])
                
                # Take top proposals
                top_k = min(100, scores.shape[0])
                _, top_idx = scores.topk(top_k)
                proposals = proposals[top_idx]
            
            # Sample proposals for ROI head training
            proposal_ious = box_iou(proposals, gt_boxes)
            max_iou_prop, matched_gt_prop = proposal_ious.max(dim=1)
            
            # Positive: IoU >= 0.5, Negative: IoU < 0.5
            pos_prop_mask = max_iou_prop >= 0.5
            neg_prop_mask = max_iou_prop < 0.5
            
            # Sample balanced batch
            num_pos = min(16, pos_prop_mask.sum().item())
            num_neg = min(48, neg_prop_mask.sum().item())
            
            if num_pos > 0:
                pos_idx = pos_prop_mask.nonzero(as_tuple=True)[0][:num_pos]
                neg_idx = neg_prop_mask.nonzero(as_tuple=True)[0][:num_neg]
                sample_idx = torch.cat([pos_idx, neg_idx])
                
                sampled_proposals = proposals[sample_idx]
                
                # Add batch index for roi_align
                batch_idx = torch.full((sampled_proposals.shape[0], 1), b, device=device, dtype=torch.float32)
                rois = torch.cat([batch_idx, sampled_proposals], dim=1)
                
                # ROI Align
                roi_features = self._roi_align(features, rois, self.roi_output_size)
                roi_features = roi_features.view(roi_features.shape[0], -1)
                
                # ROI head forward
                roi_out = self.roi_head(roi_features)
                cls_logits = self.cls_head(roi_out)
                box_deltas = self.box_head(roi_out)
                
                # ROI classification targets
                roi_labels = torch.zeros(sample_idx.shape[0], dtype=torch.long, device=device)
                roi_labels[:num_pos] = gt_labels[matched_gt_prop[pos_idx]].long()
                
                roi_cls_loss = F.cross_entropy(cls_logits, roi_labels)
                roi_cls_losses.append(roi_cls_loss)
                
                # ROI regression loss (only for positive proposals)
                if num_pos > 0:
                    pos_box_deltas = box_deltas[:num_pos]
                    pos_gt = gt_boxes[matched_gt_prop[pos_idx]]
                    pos_proposals = sampled_proposals[:num_pos]
                    target_box_deltas = self._compute_box_deltas(pos_proposals, pos_gt)
                    
                    # Get deltas for the correct class
                    pos_labels = roi_labels[:num_pos]
                    idx = torch.arange(num_pos, device=device)
                    pred_box_deltas = pos_box_deltas.view(num_pos, self.num_classes, 4)
                    pred_box_deltas = pred_box_deltas[idx, pos_labels]
                    
                    roi_reg_loss = F.smooth_l1_loss(pred_box_deltas, target_box_deltas, reduction='mean')
                    roi_reg_losses.append(roi_reg_loss)
        
        # Sum all losses
        def sum_losses(losses):
            if len(losses) == 0:
                return torch.tensor(0.0, device=device, requires_grad=True)
            return torch.stack(losses).mean()

        total_rpn_cls_loss = sum_losses(rpn_cls_losses)
        total_rpn_reg_loss = sum_losses(rpn_reg_losses)
        total_roi_cls_loss = sum_losses(roi_cls_losses)
        total_roi_reg_loss = sum_losses(roi_reg_losses)

        total_loss = (
            self.rpn_lambda * (total_rpn_cls_loss + total_rpn_reg_loss) +
            self.roi_lambda * (total_roi_cls_loss + total_roi_reg_loss)
        ) / batch_size

        return {
            'loss': total_loss,
            'loss_rpn_cls': total_rpn_cls_loss / batch_size,
            'loss_rpn_reg': total_rpn_reg_loss / batch_size,
            'loss_roi_cls': total_roi_cls_loss / batch_size,
            'loss_roi_reg': total_roi_reg_loss / batch_size,
        }
    
    def _compute_box_deltas(self, src_boxes, tgt_boxes):
        src_w = src_boxes[:, 2] - src_boxes[:, 0]
        src_h = src_boxes[:, 3] - src_boxes[:, 1]
        src_cx = src_boxes[:, 0] + 0.5 * src_w
        src_cy = src_boxes[:, 1] + 0.5 * src_h
        
        tgt_w = tgt_boxes[:, 2] - tgt_boxes[:, 0]
        tgt_h = tgt_boxes[:, 3] - tgt_boxes[:, 1]
        tgt_cx = tgt_boxes[:, 0] + 0.5 * tgt_w
        tgt_cy = tgt_boxes[:, 1] + 0.5 * tgt_h
        
        dx = (tgt_cx - src_cx) / src_w
        dy = (tgt_cy - src_cy) / src_h
        dw = torch.log(tgt_w / src_w)
        dh = torch.log(tgt_h / src_h)
        
        return torch.stack([dx, dy, dw, dh], dim=1)
    
    def _inference(self, features, rpn_cls, rpn_reg, anchors, feat_h, feat_w, device):
        batch_size = features.shape[0]
        
        rpn_cls = rpn_cls.permute(0, 2, 3, 1).contiguous().view(batch_size, -1)
        rpn_reg = rpn_reg.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 4)
        
        results = []
        
        for b in range(batch_size):
            # Generate proposals
            proposals = self._apply_box_deltas(anchors, rpn_reg[b])
            proposals = proposals.clamp(min=0, max=self.input_size)
            scores = torch.sigmoid(rpn_cls[b])
            
            # NMS on proposals
            keep = nms(proposals, scores, iou_threshold=0.7)
            keep = keep[:100]  # Top 100
            proposals = proposals[keep]
            
            if proposals.shape[0] == 0:
                results.append({'boxes': torch.empty(0, 4, device=device),
                               'scores': torch.empty(0, device=device),
                               'labels': torch.empty(0, dtype=torch.long, device=device)})
                continue
            
            # ROI Align
            batch_idx = torch.full((proposals.shape[0], 1), b, device=device, dtype=torch.float32)
            rois = torch.cat([batch_idx, proposals], dim=1)
            roi_features = self._roi_align(features, rois, self.roi_output_size)
            roi_features = roi_features.view(roi_features.shape[0], -1)
            
            # ROI head forward
            roi_out = self.roi_head(roi_features)
            cls_logits = self.cls_head(roi_out)
            box_deltas = self.box_head(roi_out)
            
            # Get predictions
            cls_probs = F.softmax(cls_logits, dim=1)
            
            all_boxes = []
            all_scores = []
            all_labels = []
            
            # Process each class (skip background class 0)
            for c in range(1, self.num_classes):
                class_scores = cls_probs[:, c]
                class_deltas = box_deltas.view(-1, self.num_classes, 4)[:, c]
                
                # Apply deltas
                class_boxes = self._apply_box_deltas(proposals, class_deltas)
                class_boxes = class_boxes.clamp(min=0, max=self.input_size)
                
                # Filter by score (lowered threshold to 0.05 to see more detections)
                mask = class_scores > 0.05
                if mask.any():
                    filtered_boxes = class_boxes[mask]
                    filtered_scores = class_scores[mask]
                    
                    # NMS per class
                    keep = nms(filtered_boxes, filtered_scores, iou_threshold=0.5)
                    all_boxes.append(filtered_boxes[keep])
                    all_scores.append(filtered_scores[keep])
                    all_labels.append(torch.full((keep.shape[0],), c, device=device, dtype=torch.long))
            
            if all_boxes:
                boxes = torch.cat(all_boxes, dim=0)
                scores = torch.cat(all_scores, dim=0)
                labels = torch.cat(all_labels, dim=0)
            else:
                boxes = torch.empty(0, 4, device=device)
                scores = torch.empty(0, device=device)
                labels = torch.empty(0, dtype=torch.long, device=device)
            
            results.append({'boxes': boxes, 'scores': scores, 'labels': labels})
        
        return results

class CustomConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(CustomConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.1, inplace=True)
    
    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))

class CustomResidualBlock(nn.Module):
    def __init__(self, channels):
        super(CustomResidualBlock, self).__init__()
        self.conv1 = CustomConvBlock(channels, channels // 2, kernel_size=1, padding=0)
        self.conv2 = CustomConvBlock(channels // 2, channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        return x + self.conv2(self.conv1(x))

#endregion
#region API

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

def load_rcnn_model(device):
    import torchvision
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features # type: ignore
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)

    return model.to(device)

def load_custom_model(device='cpu', num_classes=2, input_size=1024):
    model = CustomPedestrianDetector(
        num_classes=num_classes,
        num_anchors=9,
        input_size=input_size
    )
    return model.to(device)

#endregion