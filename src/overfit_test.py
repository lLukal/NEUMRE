"""
Overfit test: Train on a tiny subset (5 images) for many epochs.
If the model can't overfit, there's a bug in the model or training loop.
"""
import torch
import torch.nn.functional as F
from tqdm import tqdm
from data import load_citypersons_dataset, collate_fn
from models import load_custom_model
from torch.utils.data import DataLoader, Subset

def overfit_test():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load just 5 images
    full_dataset = load_citypersons_dataset(split="train")
    tiny_dataset = Subset(full_dataset, list(range(5)))
    tiny_loader = DataLoader(tiny_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
    
    print(f"Tiny dataset size: {len(tiny_dataset)} images")
    
    # Check what's in the dataset
    print("\n=== Dataset Check ===")
    for i, (images, targets) in enumerate(tiny_loader):
        for j, (img, tgt) in enumerate(zip(images, targets)):
            print(f"Image {i*2+j}: shape={img.shape}, boxes={tgt['boxes'].shape}, labels={tgt['labels']}")
            if tgt['boxes'].shape[0] > 0:
                print(f"  Box example: {tgt['boxes'][0]}")
        if i >= 1:
            break
    print("=====================\n")
    
    # Load model
    model = load_custom_model(device=device, num_classes=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # Higher LR for overfit test
    
    num_epochs = 50
    print(f"Training for {num_epochs} epochs on {len(tiny_dataset)} images...")
    print("If model is working, loss should go to near 0 and it should detect boxes.\n")
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        rpn_cls_total = 0.0
        roi_cls_total = 0.0
        num_batches = 0
        
        for images, targets in tiny_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            loss_dict = model(images, targets)
            loss = loss_dict['loss']
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            rpn_cls_total += loss_dict['loss_rpn_cls'].item()
            roi_cls_total += loss_dict['loss_roi_cls'].item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        avg_rpn = rpn_cls_total / num_batches
        avg_roi = roi_cls_total / num_batches
        
        # Every 5 epochs, test inference
        if (epoch + 1) % 5 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                # Test on first image
                test_img = tiny_dataset[0][0].unsqueeze(0).to(device)
                outputs = model(test_img)[0]
                num_boxes = outputs['boxes'].shape[0]
                max_score = outputs['scores'].max().item() if num_boxes > 0 else 0.0
            
            print(f"Epoch {epoch+1:3d} | Loss: {avg_loss:.4f} | RPN: {avg_rpn:.4f} | ROI: {avg_roi:.4f} | "
                  f"Boxes: {num_boxes}, MaxScore: {max_score:.4f}")
        else:
            print(f"Epoch {epoch+1:3d} | Loss: {avg_loss:.4f} | RPN: {avg_rpn:.4f} | ROI: {avg_roi:.4f}")
    
    # Final test
    print("\n=== Final Inference Test ===")
    model.eval()
    model._debug_inference = True  # Enable debug output
    with torch.no_grad():
        for i in range(min(3, len(tiny_dataset))):
            img, target = tiny_dataset[i]
            img = img.unsqueeze(0).to(device)
            
            # Get raw outputs before filtering
            _, scales = model._prepare_images([img.squeeze(0)])
            features = model.backbone(F.interpolate(img, size=(model.input_size, model.input_size), mode='bilinear', align_corners=False))
            feat_h, feat_w = features.shape[2], features.shape[3]
            rpn_features = F.relu(model.rpn_conv(features))
            rpn_cls = model.rpn_cls(rpn_features)
            rpn_reg = model.rpn_reg(rpn_features)
            anchors = model._generate_anchors(feat_h, feat_w, device)
            
            rpn_cls_flat = rpn_cls.permute(0, 2, 3, 1).contiguous().view(1, -1)
            rpn_scores = torch.sigmoid(rpn_cls_flat[0])
            
            print(f"\nImage {i}:")
            print(f"  Ground truth boxes: {target['boxes'].shape[0]}")
            print(f"  RPN scores - max: {rpn_scores.max():.4f}, mean: {rpn_scores.mean():.4f}")
            print(f"  RPN scores > 0.5: {(rpn_scores > 0.5).sum().item()}")
            
            # Full inference
            outputs = model(img)[0]
            print(f"  Predicted boxes: {outputs['boxes'].shape[0]}")
            if outputs['boxes'].shape[0] > 0:
                print(f"  Scores: {outputs['scores'][:5]}")
            else:
                print("  No boxes detected!")

if __name__ == "__main__":
    overfit_test()
