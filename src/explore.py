import sys
import cv2
import numpy as np
from pathlib import Path
import torch
from ultralytics import YOLO # type: ignore
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
    # import torchvision
    # model = torchvision.models.detection.ssd300_vgg16(weights=None, num_classes=2)
    # model.load_state_dict(torch.load(weights_path, map_location=device))
    # return model.to(device).eval()
    pass

def load_detr(weights_path, device):
    # Load from Torch Hub to ensure the architecture matches the citypersons/detr weights
    model = torch.hub.load('facebookresearch/detr:main', 'detr_resnet50', pretrained=False, num_classes=2)
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
    
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    scores, _ = probas.max(-1)
    bboxes = outputs['pred_boxes'][0] # cxcywh normalized
    
    # Convert to xyxy pixels
    x_c, y_c, bw, bh = bboxes.unbind(-1)
    b = [(x_c - 0.5 * bw), (y_c - 0.5 * bh), (x_c + 0.5 * bw), (y_c + 0.5 * bh)]
    pixel_boxes = torch.stack(b, dim=-1) * torch.tensor([w, h, w, h], device=device)
    
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
        elif model_type == "custom":
            model = load_custom(weights_path, device)
            outputs = run_torchvision_inference(model, device, image)
        elif model_type == "detr":
            model = load_detr(weights_path, device)
            outputs = run_detr_inference(model, device, image)
            
        print(f"{model_type} boxes:", outputs["boxes"].shape[0])
        pred_img = draw_torch_predictions(pred_img, outputs, conf=0.5)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    vis = np.hstack([gt_img, pred_img])
    if vis.shape[1] > 1200:
        scale = 1200 / vis.shape[1]
        vis = cv2.resize(vis, (int(vis.shape[1] * scale), int(vis.shape[0] * scale)))

    cv2.imshow("GT | Predictions", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

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