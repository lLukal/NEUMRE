import cv2
import numpy as np
from pathlib import Path
import torch
from ultralytics import YOLO # type: ignore

def draw_yolo_labels(image, label_path, color=(0, 255, 0)):
    """
    image: BGR image (cv2)
    label_path: path to YOLO .txt label
    """
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

def draw_yolo_predictions(image, results, conf=0.01):
    """
    image: BGR image
    results: Ultralytics YOLO result object
    """
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

def visualize_sample(
    dataset_root,
    weights_path,
    split="val",
    index=0,
    model_type="yolo",  # "yolo", "torch", or "custom"
):
    dataset_root = Path(dataset_root)
    img_dir = dataset_root / "images" / split
    lbl_dir = dataset_root / "labels" / split

    img_paths = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
    img_path = img_paths[index]
    lbl_path = lbl_dir / img_path.with_suffix(".txt").name

    image = cv2.imread(str(img_path))
    gt_img = image.copy() # type: ignore
    pred_img = image.copy() # type: ignore

    # Draw GT
    gt_img = draw_yolo_labels(gt_img, lbl_path)

    if model_type == "yolo":
        model = YOLO(weights_path)
        results = model(img_path, verbose=False, conf=0.01)[0]
        print("YOLO boxes:", 0 if results.boxes is None else len(results.boxes))
        pred_img = draw_yolo_predictions(pred_img, results, conf=0.05)

    elif model_type == "torch":
        model, device = load_torch_model(weights_path)
        outputs = run_torch_inference(model, device, image)
        print("Torch boxes:", outputs["boxes"].shape[0])
        pred_img = draw_torch_predictions(pred_img, outputs, conf=0.5)

    elif model_type == "custom":
        model, device = load_custom_model_for_inference(weights_path)
        outputs = run_custom_inference(model, device, image, debug=True)
        print("Custom boxes:", outputs["boxes"].shape[0])
        pred_img = draw_torch_predictions(pred_img, outputs, conf=0.05)  # Lower threshold

    else:
        raise ValueError("model_type must be 'yolo', 'torch', or 'custom'")

    vis = np.hstack([gt_img, pred_img])

    max_width = 1200
    h, w = vis.shape[:2]
    if w > max_width:
        scale = max_width / w
        vis = cv2.resize(vis, (int(w * scale), int(h * scale)))

    cv2.imshow("GT | Predictions", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def load_torch_model(weights_path, device="cuda"):
    import torchvision

    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # Example: Faster R-CNN
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=None,
        num_classes=2,
    )

    checkpoint = torch.load(weights_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    return model, device

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

def run_torch_inference(model, device, image):
    """
    image: BGR OpenCV image
    """
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(img_rgb).float().permute(2, 0, 1) / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)[0]

    return outputs


def load_custom_model_for_inference(weights_path, input_size=416, device="cuda"):
    """Load custom pedestrian detector for inference."""
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


def run_custom_inference(model, device, image, input_size=416, debug=True):
    """
    Run inference with custom model.
    image: BGR OpenCV image
    """
    orig_h, orig_w = image.shape[:2]
    
    # Resize to model input size
    img_resized = cv2.resize(image, (input_size, input_size))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
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
    if outputs['boxes'].shape[0] > 0:
        scale_x = orig_w / input_size
        scale_y = orig_h / input_size
        outputs['boxes'][:, 0] *= scale_x
        outputs['boxes'][:, 2] *= scale_x
        outputs['boxes'][:, 1] *= scale_y
        outputs['boxes'][:, 3] *= scale_y
    
    return outputs

def debug_check_labels(dataset_root, split="train", num_samples=3):
    """Check if labels are correct by printing box info."""
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


if __name__ == '__main__':
    # First check if labels are correct
    debug_check_labels("../data/yolo/citypersons", split="val", num_samples=3)
    
    # Then try visualization
    visualize_sample("../data/yolo/citypersons", "./trained_models/custom_best.pth", model_type="custom")
