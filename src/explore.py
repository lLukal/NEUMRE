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

def draw_yolo_predictions(image, results, conf=0.25):
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
    model_type="yolo",  # "yolo" or "torch"
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

    else:
        raise ValueError("model_type must be 'yolo' or 'torch'")

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

if __name__ == '__main__':
    visualize_sample(
        dataset_root="../data/yolo/penn_fudan",
        weights_path="../trained_models/yolo_penn_fudan.pt",
        split="val",
        index=1,
        model_type="yolo",
    )
