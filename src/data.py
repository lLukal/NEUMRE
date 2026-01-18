import os
import cv2
from scipy.io import loadmat
import numpy as np
from torch.utils.data import DataLoader
from utils import *

#region PRIVATE

def read_seq(seq_path, n):
    cap = cv2.VideoCapture(seq_path)
    frames = []

    frame_id = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % n == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append((frame_id, frame))

        frame_id += 1

    cap.release()
    return frames

def read_vbb(vbb_path):
    mat = loadmat(vbb_path)
    A = mat["A"][0][0]

    obj_lists = A[1][0]  # frame-wise annotations
    annotations = {}

    for frame_id, frame_objs in enumerate(obj_lists):
        boxes = []

        # --- CRITICAL FIX ---
        # frame_objs is often shape (1,), containing a list
        if frame_objs.size == 0:
            annotations[frame_id] = np.zeros((0, 4), dtype=np.float32)
            continue

        if isinstance(frame_objs[0], (list, tuple, np.ndarray)):
            objects = frame_objs[0]
        else:
            objects = frame_objs

        for obj in objects:
            # obj is now a tuple of length 5
            # obj[1] = bbox [x, y, w, h] shape (1,4)
            pos = np.array(obj[1]).squeeze()

            if pos.shape != (4,):
                continue

            x, y, w, h = pos
            boxes.append([x, y, x + w, y + h])

        annotations[frame_id] = np.array(boxes, dtype=np.float32)

    return annotations

def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)

def detr_collate_fn(batch):
    images, targets = zip(*batch)

    # stack images
    images = torch.stack(images)

    detr_targets = []

    for img, target in zip(images, targets):
        h, w = img.shape[1:]

        boxes = target["boxes"].clone()

        if boxes.numel() > 0:
            # convert x1,y1,x2,y2 → cx,cy,w,h
            x1, y1, x2, y2 = boxes.unbind(1)
            cx = (x1 + x2) / 2 / w
            cy = (y1 + y2) / 2 / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            boxes = torch.stack([cx, cy, bw, bh], dim=1)

        detr_targets.append({
            "class_labels": target["labels"],
            "boxes": boxes
        })

    return images, detr_targets

def load_caltech_dataset(split="train"):
    dataset = DLDataset("../data/yolo/caltech", split=split)
    return dataset

def load_citypersons_dataset(split="train"):
    dataset = DLDataset("../data/yolo/citypersons", split=split)
    return dataset

def load_penn_fudan_dataset(split="train"):
    dataset = DLDataset("../data/yolo/penn_fudan", split=split)
    return dataset

#endregion
#region API

def load_dataset(dataset_type: DatasetType, model_type: ModelType = ModelType.YOLO, split = 'train'):
    print(f'\tLoading dataset (split: {split})...')
    dataset = None

    if dataset_type == DatasetType.CALTECH:
        dataset = load_caltech_dataset(split)
    elif dataset_type == DatasetType.CITYPERSONS:
        dataset = load_citypersons_dataset(split)
    elif dataset_type == DatasetType.PENN_FUDAN:
        dataset = load_penn_fudan_dataset(split)
    else:
        raise AttributeError('Invalid Dataset Type')
    
    if model_type != ModelType.DETR:
        loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
    else:
        loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=detr_collate_fn)
    
    return loader
    
#endregion