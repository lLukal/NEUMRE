import cv2
import numpy as np
from pathlib import Path
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

def visualize_yolo_sample(
    dataset_root,
    weights_path,
    split="val",
    index=0,
):
    dataset_root = Path(dataset_root)
    img_dir = dataset_root / "images" / split
    lbl_dir = dataset_root / "labels" / split

    img_paths = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
    img_path = img_paths[index]
    lbl_path = lbl_dir / img_path.with_suffix(".txt").name

    # Load image
    image = cv2.imread(str(img_path))
    gt_img = image.copy() # type: ignore
    pred_img = image.copy() # type: ignore

    # Draw ground truth
    gt_img = draw_yolo_labels(gt_img, lbl_path)

    # Load model + predict
    model = YOLO(weights_path)
    results = model(img_path, verbose=False, conf=0.01)[0]
    print(results.boxes)
    print("num boxes:", 0 if results.boxes is None else len(results.boxes))

    # Draw predictions
    pred_img = draw_yolo_predictions(pred_img, results)

    # Stack side by side
    vis = np.hstack([gt_img, pred_img])

    max_width = 1200
    h, w = vis.shape[:2]
    if w > max_width:
        scale = max_width / w
        vis = cv2.resize(vis, (int(w * scale), int(h * scale)))

    cv2.imshow("GT | YOLO Predictions", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    visualize_yolo_sample(
        dataset_root="../data/yolo/citypersons",
        weights_path="runs/detect/train/weights/best.pt",
        split="val",
        index=4,
    )
