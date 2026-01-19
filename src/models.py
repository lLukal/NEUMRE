#region PRIVATE

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.ops import nms, box_iou

#endregion
#region API

from utils import *


class CustomPedestrianDetector(nn.Module):
    """
    Lightweight Faster R-CNN-style model for pedestrian detection.
    
    Architecture:
        - Backbone: Lightweight CNN feature extractor
        - RPN: Region Proposal Network for generating object proposals
        - ROI Head: ROI Align + classification and box regression heads
    
    Compatible with PyTorch DataLoader returning (images, targets) where:
        - images: list of tensors [C, H, W]
        - targets: list of dicts with 'boxes' [N, 4] and 'labels' [N]
    """
    
    def __init__(self, num_classes=2, num_anchors=9, input_size=416):
        """
        Args:
            num_classes: Number of classes (default 2: background + pedestrian)
            num_anchors: Number of anchor boxes per grid cell for RPN
            input_size: Expected input image size (square) - reduced to 416 for faster training
        """
        super(CustomPedestrianDetector, self).__init__()
        
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.input_size = input_size
        self.feature_channels = 128  # Reduced from 256
        self.roi_output_size = 7
        
        # Anchor settings for RPN (optimized for pedestrians)
        self.anchor_ratios = [0.5, 1.0, 2.0]
        self.anchor_scales = [16, 32, 64]  # Smaller scales for smaller input
        
        # Backbone: Lightweight feature extractor
        self.backbone = self._make_backbone()
        
        # Region Proposal Network (RPN)
        self.rpn_conv = nn.Conv2d(self.feature_channels, 256, kernel_size=3, padding=1)
        self.rpn_cls = nn.Conv2d(256, num_anchors, kernel_size=1)  # objectness
        self.rpn_reg = nn.Conv2d(256, num_anchors * 4, kernel_size=1)  # box deltas
        
        # ROI Head: classification and box regression (simplified)
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
        """Create simple CNN backbone - much simpler for faster training."""
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
        """
        Forward pass (Faster R-CNN style).
        
        Args:
            images: list of tensors [C, H, W] or batched tensor [B, C, H, W]
            targets: list of dicts with 'boxes' and 'labels' (training only)
            
        Returns:
            Training: dict of losses
            Inference: list of dicts with 'boxes', 'scores', 'labels'
        """
        scales = None
        if isinstance(images, list):
            images, scales = self._prepare_images(images)
        
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
                anchors, targets, feat_h, feat_w, device, scales
            )
        else:
            return self._inference(
                features, rpn_cls_logits, rpn_box_deltas,
                anchors, feat_h, feat_w, device
            )
    
    def _prepare_images(self, images):
        """Prepare list of images into batched tensor and return scale factors."""
        processed = []
        scales = []
        for img in images:
            _, orig_h, orig_w = img.shape
            scale_x = self.input_size / orig_w
            scale_y = self.input_size / orig_h
            scales.append((scale_x, scale_y))
            
            img = F.interpolate(
                img.unsqueeze(0), 
                size=(self.input_size, self.input_size), 
                mode='bilinear', 
                align_corners=False
            )
            processed.append(img)
        return torch.cat(processed, dim=0), scales
    
    def _generate_anchors(self, feat_h, feat_w, device):
        """Generate anchor boxes for RPN."""
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
        """Apply predicted deltas to anchors to get proposals."""
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
        """Simple ROI Align implementation."""
        from torchvision.ops import roi_align
        # boxes should be [N, 5] with batch index, but we handle single batch
        return roi_align(features, boxes, output_size, spatial_scale=1.0/16.0)
    
    def _compute_loss(self, features, rpn_cls, rpn_reg, anchors, targets, feat_h, feat_w, device, scales=None):
        """Compute Faster R-CNN losses."""
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
            gt_boxes = target['boxes'].clone()
            gt_labels = target['labels']
            
            if gt_boxes.shape[0] == 0:
                continue
            
            # Scale GT boxes to input_size coordinates if scales provided
            if scales is not None:
                scale_x, scale_y = scales[b]
                gt_boxes[:, 0] *= scale_x  # x1
                gt_boxes[:, 2] *= scale_x  # x2
                gt_boxes[:, 1] *= scale_y  # y1
                gt_boxes[:, 3] *= scale_y  # y2
            
            # Clamp boxes to valid range
            gt_boxes = gt_boxes.clamp(min=0, max=self.input_size)
            
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
            
            # Positive: IoU >= 0.3 (lowered from 0.5), Negative: IoU < 0.3
            pos_prop_mask = max_iou_prop >= 0.3
            neg_prop_mask = max_iou_prop < 0.3
            
            # Sample balanced batch - equal pos/neg to avoid class imbalance
            num_pos = min(32, pos_prop_mask.sum().item())
            num_neg = min(num_pos, neg_prop_mask.sum().item())  # Match neg to pos count
            
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
                
                # Class weights: penalize missing pedestrians more than false positives
                class_weights = torch.tensor([1.0, 3.0], device=device)  # bg=1, ped=3
                roi_cls_loss = F.cross_entropy(cls_logits, roi_labels, weight=class_weights)
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
        
        # Sum all losses (if empty, use torch.tensor(0.0, device=device, requires_grad=True))
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
        """Compute box regression deltas."""
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
        """Inference: generate final detections."""
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
            keep = keep[:100]  # Top 100 proposals
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
                
                # Filter by score - only keep high confidence detections (>0.9)
                mask = class_scores > 0.9
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


class ConvBlock(nn.Module):
    """Convolutional block with BatchNorm and LeakyReLU."""
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.1, inplace=True)
    
    def forward(self, x):
        return self.activation(self.bn(self.conv(x)))


class ResidualBlock(nn.Module):
    """Residual block with two convolutions."""
    
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvBlock(channels, channels // 2, kernel_size=1, padding=0)
        self.conv2 = ConvBlock(channels // 2, channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        return x + self.conv2(self.conv1(x))


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

def load_custom_model(device='cpu', num_classes=2, input_size=416):
    """
    Load custom pedestrian detection model.
    
    Args:
        device: Device to load model on ('cpu' or 'cuda')
        num_classes: Number of classes (default 2: background + pedestrian)
        input_size: Input image size (default 416 for faster training)
        
    Returns:
        CustomPedestrianDetector model on specified device
    """
    model = CustomPedestrianDetector(
        num_classes=num_classes,
        num_anchors=9,
        input_size=input_size
    )
    return model.to(device)

def load_rcnn_model(device):
    import torchvision
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features # type: ignore
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)

    return model.to(device)

#endregion