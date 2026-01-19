import sys
import cv2
import numpy as np
from pathlib import Path
import torch
from transformers import DetrForObjectDetection
from ultralytics import YOLO # type: ignore
from models import CustomPedestrianDetector
from utils import DatasetType, ModelType # type: ignore

#region DRAW

def draw_yolo_labels(image, label_path, color=(0, 255, 0)):
    h, w = image.shape[:2]

    if not label_path.exists():
        return image

    with open(label_path) as f:
        for line in f:
            cls, cx, cy, bw, bh = map(float, line.strip().split())

            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)

            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                image,
                "GT",
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )

    return image

def draw_yolo_predictions(image, results, conf=0.25):
    for box in results.boxes:
        if box.conf < conf:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        score = float(box.conf)

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            image,
            f"Pred {score:.2f}",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
        )

    return image

# def visualize_sample(
#     dataset_root,
#     weights_path,
#     split="val",
#     index=0,
#     model_type="yolo",  # "yolo", "torch", or "custom"
# ):
#     dataset_root = Path(dataset_root)
#     img_dir = dataset_root / "images" / split
#     lbl_dir = dataset_root / "labels" / split

#     img_paths = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
#     img_path = img_paths[index]
#     lbl_path = lbl_dir / img_path.with_suffix(".txt").name

#     image = cv2.imread(str(img_path))
#     gt_img = image.copy() # type: ignore
#     pred_img = image.copy() # type: ignore

#     # Draw GT
#     gt_img = draw_yolo_labels(gt_img, lbl_path)

#     if model_type == "yolo":
#         model = YOLO(weights_path)
#         results = model(img_path, verbose=False, conf=0.01)[0]
#         print("YOLO boxes:", 0 if results.boxes is None else len(results.boxes))
#         pred_img = draw_yolo_predictions(pred_img, results, conf=0.05)

#     elif model_type == "torch":
#         model, device = load_torch_model(weights_path)
#         outputs = run_torch_inference(model, device, image)
#         print("Torch boxes:", outputs["boxes"].shape[0])
#         pred_img = draw_torch_predictions(pred_img, outputs, conf=0.5)

    # elif model_type == "custom":
    #     model, device = load_custom_model_for_inference(weights_path)
    #     outputs = run_custom_inference(model, device, image, debug=True)
    #     print("Custom boxes:", outputs["boxes"].shape[0])
    #     pred_img = draw_torch_predictions(pred_img, outputs, conf=0.9)  # High confidence only

#     else:
#         raise ValueError("model_type must be 'yolo', 'torch', or 'custom'")

#     vis = np.hstack([gt_img, pred_img])

#     max_width = 1200
#     h, w = vis.shape[:2]
#     if w > max_width:
#         scale = max_width / w
#         vis = cv2.resize(vis, (int(w * scale), int(h * scale)))

#     cv2.imshow("GT | Predictions", vis)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

def draw_torch_predictions(image, outputs, conf=0.25):
    boxes = outputs.get("boxes")
    scores = outputs.get("scores")

    if boxes is None or scores is None:
        return image

    boxes = boxes.cpu().numpy()
    scores = scores.cpu().numpy()

    for (x1, y1, x2, y2), score in zip(boxes, scores):
        if score < conf:
            continue

        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        
        # Green for high confidence (>0.9), red for lower
        color = (0, 255, 0) if score > 0.9 else (0, 0, 255)

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            image,
            f"Pred {score:.2f}",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
        )

    return image


def draw_torch_predictions_high_only(image, outputs, conf=0.9):
    """Draw only high confidence predictions (green boxes only)."""
    boxes = outputs.get("boxes")
    scores = outputs.get("scores")

    if boxes is None or scores is None:
        return image

    boxes = boxes.cpu().numpy()
    scores = scores.cpu().numpy()

    for (x1, y1, x2, y2), score in zip(boxes, scores):
        if score < conf:
            continue

        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        color = (0, 255, 0)  # Green only

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            image,
            f"Pred {score:.2f}",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
        )

    return image

# def load_torch_model(weights_path, device="cuda"):
#     import torchvision

#     device = torch.device(device if torch.cuda.is_available() else "cpu")

#     # Example: Faster R-CNN
#     model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
#         weights=None,
#         num_classes=2,
#     )

#     checkpoint = torch.load(weights_path, map_location=device)
#     model.load_state_dict(checkpoint)
#     model.to(device)
#     model.eval()

#     return model, device

#endregion
#region MODEL

def load_yolo(weights_path):
    return YOLO(weights_path), torch.device("cpu")

def load_faster_rcnn(weights_path, device):
    import torchvision
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None, num_classes=2)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    return model.to(device).eval()

def load_custom(weights_path, device):
    model = CustomPedestrianDetector(
        num_classes=2,
        num_anchors=9,
        input_size=1024
    )
    checkpoint = torch.load(weights_path, map_location=device)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    cleaned_state_dict = {}
    for k, v in state_dict.items():
        cleaned_state_dict[k.replace("module.", "")] = v

    model.load_state_dict(cleaned_state_dict, strict=True)

    return model.to(device).eval()

def load_detr(weights_path, device):
    model = DetrForObjectDetection.from_pretrained(
        "facebook/detr-resnet-50",
        num_labels=1, 
        ignore_mismatched_sizes=True 
    )
    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint['model'] if 'model' in checkpoint else checkpoint, strict=False) # type: ignore
    return model.to(device).eval() # type: ignore

def run_torchvision_inference(model, device, image):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb).float().permute(2, 0, 1) / 255.0
    with torch.no_grad():
        return model([img_tensor.to(device)])[0]

def run_detr_inference(model, device, image):
    h, w = image.shape[:2]
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb).float().permute(2, 0, 1) / 255.0
    # DETR specific normalization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_tensor = (img_tensor - mean) / std
    
    with torch.no_grad():
        outputs = model(img_tensor.unsqueeze(0).to(device))
    
    logits = outputs.logits          # (1, Q, C+1)
    boxes = outputs.pred_boxes       # (1, Q, 4) in cxcywh normalized

    probs = logits.softmax(-1)[0, :, :-1]  # remove "no-object"
    scores, labels = probs.max(-1)

    keep = scores > 0.5
    scores = scores[keep]
    boxes = boxes[0, keep]

    # Convert cxcywh -> xyxy (pixel coords)
    cx, cy, bw, bh = boxes.unbind(-1)
    x1 = (cx - bw / 2) * w
    y1 = (cy - bh / 2) * h
    x2 = (cx + bw / 2) * w
    y2 = (cy + bh / 2) * h

    pixel_boxes = torch.stack([x1, y1, x2, y2], dim=-1)
    
    return {"boxes": pixel_boxes, "scores": scores}

#endregion
#region VISUALIZE

def visualize_sample(dataset_root, weights_path, split="val", index=0, model_type="yolo"):
    dataset_root = Path(dataset_root)
    img_paths = sorted(list((dataset_root/"images"/split).glob("*.jpg")) + list((dataset_root/"images"/split).glob("*.png")))
    img_path = img_paths[index]
    lbl_path = dataset_root/"labels"/split/img_path.with_suffix(".txt").name

    image = cv2.imread(str(img_path))
    gt_img, pred_img = image.copy(), image.copy() # type: ignore
    gt_img = draw_yolo_labels(gt_img, lbl_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_type == "yolo":
        model, _ = load_yolo(weights_path)
        results = model(img_path, verbose=False, conf=0.01)[0]
        pred_img = draw_yolo_predictions(pred_img, results, conf=0.05)
    
    elif model_type in ["rcnn", "custom", "detr"]:
        if model_type == "rcnn":
            model = load_faster_rcnn(weights_path, device)
            outputs = run_torchvision_inference(model, device, image)
            print(f"{model_type} boxes:", outputs["boxes"].shape[0])
            pred_img = draw_torch_predictions(pred_img, outputs, conf=0.5)
        elif model_type == "detr":
            model = load_detr(weights_path, device)
            outputs = run_detr_inference(model, device, image)
            print(f"{model_type} boxes:", outputs["boxes"].shape[0])
            pred_img = draw_torch_predictions(pred_img, outputs, conf=0.5)
        elif model_type == "custom":
            model, device = load_custom_model_for_inference(weights_path)
            outputs = run_custom_inference(model, device, image, debug=True)
            # print("Custom boxes:", outputs["boxes"].shape[0])
            pred_img = draw_torch_predictions(pred_img, outputs, conf=0.9)  # High confidence only

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    vis = np.hstack([gt_img, pred_img])
    if vis.shape[1] > 1200:
        scale = 1200 / vis.shape[1]
        vis = cv2.resize(vis, (int(vis.shape[1] * scale), int(vis.shape[0] * scale)))

    cv2.imshow("GT | Predictions", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def load_custom_model_for_inference(weights_path, input_size=1024, device="cuda"):
    from models import CustomPedestrianDetector
    
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    
    model = CustomPedestrianDetector(
        num_classes=2,
        num_anchors=9,
        input_size=input_size
    )
    
    checkpoint = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    
    return model, device

def run_custom_inference(model, device, image, input_size=1024, debug=True):
    orig_h, orig_w = image.shape[:2]
    input_size = model.input_size
    
    # Resize to model input size
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)[0]
    
    if debug:
        print(f"\n=== DEBUG: Raw Model Outputs ===")
        print(f"Number of boxes: {outputs['boxes'].shape[0]}")
        if outputs['boxes'].shape[0] > 0:
            print(f"Boxes (first 5): {outputs['boxes'][:5]}")
            print(f"Scores (first 5): {outputs['scores'][:5]}")
            print(f"Labels (first 5): {outputs['labels'][:5]}")
            print(f"Max score: {outputs['scores'].max().item():.4f}")
            print(f"Min score: {outputs['scores'].min().item():.4f}")
        else:
            print("No boxes detected!")
        print("================================\n")
    
    # Scale boxes back to original image size for visualization
    if outputs['boxes'].numel() > 0:
        scale_x = orig_w / input_size
        scale_y = orig_h / input_size

        boxes = outputs['boxes']
        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y
        outputs['boxes'] = boxes
    
    return outputs

def debug_check_labels(dataset_root, split="train", num_samples=3):
    dataset_root = Path(dataset_root)
    img_dir = dataset_root / "images" / split
    lbl_dir = dataset_root / "labels" / split

    img_paths = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
    
    print(f"\n=== DEBUG: Checking {split} labels ===")
    print(f"Found {len(img_paths)} images in {img_dir}")
    
    for i in range(min(num_samples, len(img_paths))):
        img_path = img_paths[i]
        lbl_path = lbl_dir / img_path.with_suffix(".txt").name
        
        print(f"\nImage {i}: {img_path.name}")
        print(f"Label exists: {lbl_path.exists()}")
        
        if lbl_path.exists():
            with open(lbl_path) as f:
                lines = f.readlines()
                print(f"Number of objects: {len(lines)}")
                for j, line in enumerate(lines[:3]):  # Show first 3 boxes
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls, cx, cy, bw, bh = parts
                        print(f"  Box {j}: class={cls}, cx={cx}, cy={cy}, w={bw}, h={bh}")
    print("================================\n")

#endregion
#region MAIN

if __name__ == '__main__':
    import torch
    torch.cuda.empty_cache()
    import gc
    gc.collect()

    # parse system parameters
    args = sys.argv
    if len(args) < 4:
        print('Usage: python train.py <dataset> <model> <weights_path> [<split>] [<sample_index>]')
        print(f'<dataset> - options:\n\t\t' + '\n\t\t'.join([d.value for d in DatasetType]))
        print(f'<model> - options:\n\t\t' + '\n\t\t'.join([m.value for m in ModelType]))
        print(f'<weights_path> - filepath to weights')
        print(f'[<split>] - dataset split to use')
        print(f'[<sample_index>] - index of the sample in the split')
        exit(1)

    if len(args) >= 5:
        split=args[4]
    else:
        split='val'
    if len(args) >= 6:
        index=int(args[5])
    else:
        index=1

    # run
    visualize_sample(
        dataset_root=f'../data/yolo/{args[1]}', # "../data/yolo/penn_fudan"
        weights_path=args[3], # "../trained_models/detr_citypersons.pth"
        split=split,
        index=index,
        model_type=args[2]
    )

    # First check if labels are correct
    # debug_check_labels("../data/yolo/citypersons", split="val", num_samples=3)
    
    # Save first 30 images
    # dataset_root = Path("../data/yolo/citypersons")
    # weights_path = "./trained_models/custom_last.pth"
    # split = "val"
    # output_dir = Path("./visualization_output")
    # output_dir.mkdir(exist_ok=True)
    # (output_dir / "all_boxes").mkdir(exist_ok=True)
    # (output_dir / "high_conf_only").mkdir(exist_ok=True)
    
    # img_dir = dataset_root / "images" / split
    # lbl_dir = dataset_root / "labels" / split
    # img_paths = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
    
    # # Load model once
    # model, device = load_custom_model_for_inference(weights_path)
    
    # num_images = min(30, len(img_paths))
    # print(f"\nSaving {num_images} images to {output_dir}...\n")
    
    # for i in range(num_images):
    #     img_path = img_paths[i]
    #     lbl_path = lbl_dir / img_path.with_suffix(".txt").name
        
    #     image = cv2.imread(str(img_path))
    #     gt_img = image.copy() # type: ignore
    #     pred_img_all = image.copy() # type: ignore
    #     pred_img_high = image.copy() # type: ignore
        
    #     # Draw GT on left side
    #     gt_img = draw_yolo_labels(gt_img, lbl_path)
        
    #     # Get predictions
    #     outputs = run_custom_inference(model, device, image, debug=False)
        
    #     # Draw ALL boxes (green >0.9, red <0.9)
    #     pred_img_all = draw_torch_predictions(pred_img_all, outputs, conf=0.0)  # Show all
        
    #     # Draw ONLY high confidence boxes (>0.9, green only)
    #     pred_img_high = draw_torch_predictions_high_only(pred_img_high, outputs, conf=0.9)
        
    #     # Combine GT | Predictions
    #     vis_all = np.hstack([gt_img, pred_img_all])
    #     vis_high = np.hstack([gt_img, pred_img_high])
        
    #     # Resize if too wide
    #     max_width = 1600
    #     h, w = vis_all.shape[:2]
    #     if w > max_width:
    #         scale = max_width / w
    #         vis_all = cv2.resize(vis_all, (int(w * scale), int(h * scale)))
    #         vis_high = cv2.resize(vis_high, (int(w * scale), int(h * scale)))
        
    #     # Save
    #     filename = f"{i+1:03d}_{img_path.stem}.jpg"
    #     cv2.imwrite(str(output_dir / "all_boxes" / filename), vis_all)
    #     cv2.imwrite(str(output_dir / "high_conf_only" / filename), vis_high)
        
    #     print(f"Saved {i+1}/{num_images}: {filename}")
    
    # print(f"\nDone! Images saved to:")
    # print(f"  - All boxes (green>0.9, red<0.9): {output_dir / 'all_boxes'}")
    # print(f"  - High confidence only (>0.9):    {output_dir / 'high_conf_only'}")
