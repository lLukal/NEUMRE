from enum import Enum
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

#region PRIVATE

#endregion
#region API

class DatasetType(Enum):
    CALTECH = 'caltech'
    CITYPERSONS = 'citypersons'
    PENN_FUDAN = 'penn_fudan'

class ModelType(Enum):
    YOLO = 'yolo'
    RCNN = 'rcnn'
    DETR = 'detr'
    CUSTOM = 'custom'

class DLDataset(Dataset):
    """
    Generic PyTorch Dataset for YOLO-format datasets (images + labels)
    Returns:
        image: tensor CxHxW
        target: dict with 'boxes' (Nx4), 'labels' (N,)
    """

    def __init__(self, root_dir, split="train", transforms=None):
        """
        root_dir: path to dataset root (should contain images/ and labels/)
        split: 'train', 'val', or 'test'
        transforms: optional image transforms
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transforms = transforms

        self.img_dir = self.root_dir / "images" / split
        self.lbl_dir = self.root_dir / "labels" / split

        # List all images
        self.img_paths = sorted([p for p in self.img_dir.glob("*.jpg")] + [p for p in self.img_dir.glob("*.png")])

        if split != "test":
            assert self.lbl_dir.exists(), f"Label directory {self.lbl_dir} does not exist"

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert("RGB")
        w, h = image.size

        # load labels
        if self.split == "test":
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            lbl_path = self.lbl_dir / img_path.name.replace(".jpg", ".txt").replace(".png", ".txt")
            boxes_list = []
            labels_list = []

            if lbl_path.exists():
                with open(lbl_path) as f:
                    for line in f.readlines():
                        parts = line.strip().split()
                        if len(parts) != 5:
                            continue
                        cls, cx, cy, bw, bh = map(float, parts)
                        # convert YOLO normalized cx,cy,w,h to x1,y1,x2,y2
                        x1 = (cx - bw/2) * w
                        y1 = (cy - bh/2) * h
                        x2 = (cx + bw/2) * w
                        y2 = (cy + bh/2) * h
                        boxes_list.append([x1, y1, x2, y2])
                        # IMPORTANT: Add 1 to class because class 0 = background in detection models
                        # YOLO labels use 0 for pedestrian, but Faster R-CNN needs 1 for pedestrian
                        labels_list.append(int(cls) + 1)

            if len(boxes_list) == 0:
                boxes = torch.zeros((0, 4), dtype=torch.float32)
                labels = torch.zeros((0,), dtype=torch.int64)
            else:
                boxes = torch.tensor(boxes_list, dtype=torch.float32)
                labels = torch.tensor(labels_list, dtype=torch.int64)


        if self.transforms:
            image = self.transforms(image)

        # convert PIL image to tensor CxHxW
        image = torch.from_numpy(np.array(image)).permute(2,0,1).float() / 255.0

        target = {
            "boxes": boxes,
            "labels": labels,
        }

        return image, target
    
#endregion