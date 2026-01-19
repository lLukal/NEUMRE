import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader, Dataset
from torchmetrics.detection.mean_ap import MeanAveragePrecision 
import cv2
import numpy as np
import os
import time
from datetime import timedelta
import csv


class FasterCnnDataset(Dataset):
    def __init__(self, image_root_dir, labels_root_dir, images):
        self.allImgs = images
        self.imgs = self.allImgs
        self.image_root_dir = image_root_dir
        self.labels_root_dir = labels_root_dir

    
    def shuffle(self, newSize):
        if newSize >= 0 and newSize < len(self.allImgs):
            self.imgs = np.random.choice(self.allImgs, size=newSize, replace=False).tolist()


    def editImage(self, image):
        return image


    def __getitem__(self, idx):
        img_path = os.path.join(self.image_root_dir, self.imgs[idx])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)

        img /= 255.0
        img = self.editImage(img)
        image_h, image_w = img.shape[:2]
      
        img = torch.as_tensor(img).permute(2, 0, 1)

        label = os.path.join(self.labels_root_dir, self.imgs[idx].replace(".png", ".txt").replace(".jpg", ".txt"))
        with open(label, "r") as f:
            labelData = f.read()

        if len(labelData) <= 1:
            target = {}
            target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
            target["labels"] = torch.zeros((0,), dtype=torch.int64)
            target["area"] = torch.zeros((0,), dtype=torch.float32)
            target["iscrowd"] = torch.zeros((0,), dtype=torch.int64)
            target["image_id"] = torch.tensor([idx])
            return img, target
        
        lines = labelData.split("\n")[:-1]
        rectData = [list(map(float, i.split(" "))) for i in lines]

        boxes = []
        for i in rectData:
            topLeft = [i[1] - i[3] / 2, i[2] - i[4] / 2]
            bottomRight = [i[1] + i[3] / 2, i[2] + i[4] / 2]
            topLeft[0] *= image_w
            topLeft[1] *= image_h
            bottomRight[0] *= image_w
            bottomRight[1] *= image_h

            boxes.append([topLeft[0], topLeft[1], bottomRight[0], bottomRight[1]])
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 0::2] = boxes[:, 0::2].clamp(0, image_w)
        boxes[:, 1::2] = boxes[:, 1::2].clamp(0, image_h)

        labels = torch.ones((len(lines),), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = (target["boxes"][:, 3] - target["boxes"][:, 1]) * (target["boxes"][:, 2] - target["boxes"][:, 0])
        target["iscrowd"] = torch.zeros((len(lines),), dtype=torch.int64)
        target["image_id"] = torch.tensor([idx])

        return img, target

    def __len__(self):
        return len(self.imgs)



class PennFudanDataset(FasterCnnDataset):
    def __init__(self, image_root_dir, labels_root_dir):
        allImgs = []
        invalidImages = []
        
        for curImg in os.listdir(image_root_dir):
            if os.path.exists(os.path.join(labels_root_dir, curImg.replace(".png", ".txt").replace(".jpg", ".txt"))):
                allImgs.append(curImg)
            else:
                invalidImages.append(curImg)

        if len(invalidImages) > 0:
            print(f"Found {len(invalidImages)} invalid images")

        super().__init__(image_root_dir, labels_root_dir, allImgs)
     


class CityPersonsModel(FasterCnnDataset):
    def __init__(self, image_root_dir, labels_root_dir):
        allImgs = []
        invalidImages = []
        
        for curCity in os.listdir(image_root_dir):
            imgsInCity = os.listdir(os.path.join(image_root_dir, curCity))
            for curImg in imgsInCity:
                partialPath = os.path.join(curCity, curImg)
                if os.path.exists(os.path.join(labels_root_dir, partialPath.replace(".png", ".txt").replace(".jpg", ".txt"))):
                    allImgs.append(partialPath)
                else:
                    invalidImages.append(partialPath)

        if len(invalidImages) > 0:
            print(f"Found {len(invalidImages)} invalid images")

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.image_h = 800

        super().__init__(image_root_dir, labels_root_dir, allImgs)

    
    def editImage(self, img):
        org_h, org_w = img.shape[:2]
        image_w = int(round(org_w / org_h * self.image_h))
        img = cv2.resize(img, (image_w, self.image_h))

        img = (img - self.mean) / self.std
        return img



def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def collate_fn(batch):
    return tuple(zip(*batch))


def validate(model, val_loader, device):
    metric = MeanAveragePrecision()
    val_loss = 0

    with torch.no_grad():
        for images, targets in val_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            model.train()
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            val_loss += losses.item()

            model.eval()
            outputs = model(images)

            res = [{k: v.to(torch.device('cpu')) for k, v in t.items()} for t in outputs]
            gt = [{k: v.to(torch.device('cpu')) for k, v in t.items()} for t in targets]
            metric.update(res, gt)

    map_results = metric.compute()
    return (val_loss / len(val_loader)), map_results


def train(model, train_loader, device):
    model.train()
    train_loss = 0
    
    for images, targets in train_loader:
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        train_loss += losses.item()
    
    return train_loss / len(train_loader)


if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    pin_memory = device.type == "cuda"
    num_classes = 2
    epochs_without_improvement = 0
    best_map = -1

    max_num_epochs = 50
    patience = 5
    batch_size = 1
    num_workers = 3
    validate_size = 1000
    log_file = "training_log.csv"

    imagePathTrain = "../data/yolo/penn_fudan/images/train"
    labelPathTrain = "../data/yolo/penn_fudan/labels/train"
    dataset_train = PennFudanDataset(imagePathTrain, labelPathTrain)

    imagePathVal = "../data/yolo/penn_fudan/images/val"
    labelPathVal = "../data/yolo/penn_fudan/labels/val"
    dataset_validate = PennFudanDataset(imagePathVal, labelPathVal)

    with open(log_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Train_Loss", "Val_Loss", "mAP", "mAP_50", "Learning_Rate", "Time_For_Epoch"])

    model = get_model(num_classes)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=4, gamma=0.1
    )

    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn, pin_memory=pin_memory)
    dataset_validate.shuffle(validate_size)
    validate_loader = DataLoader(dataset_validate, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn, pin_memory=pin_memory)

    print(f"Starting training on {device}...")
    
    for epoch in range(max_num_epochs):
        start_time = time.time()
        avg_train_loss = train(model, train_loader, device)
        validate_start = time.time()
        avg_val_loss, map_scores = validate(model, validate_loader, device)
        end_time = time.time()
        lr_scheduler.step()

        time_for_train = validate_start - start_time  
        time_for_validate = end_time - validate_start

        td_train = timedelta(seconds=time_for_train)
        td_val = timedelta(seconds=time_for_validate)
        td_all = timedelta(seconds=time_for_train + time_for_validate)

        current_map = map_scores['map'].item()
        current_map_50 = map_scores['map_50'].item()

        print(
        f"Epoch [{epoch+1}/{max_num_epochs}] | "
        f"Train Loss: {avg_train_loss:.4f} | "
        f"Val Loss: {avg_val_loss:.4f} | "
        f"mAP: {current_map:.4f} | "
        f"mAP@50: {current_map_50:.4f} | "
        f"Time for training: {td_train} | "
        f"Time for validation: {td_val} | "
        f"Total time: {td_all}"
        )

        with open(log_file, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    epoch + 1, 
                    f"{avg_train_loss:.4f}", 
                    f"{avg_val_loss:.4f}",
                    f"{current_map:.4f}",
                    f"{current_map_50:.4f}",
                    f"{optimizer.param_groups[0]['lr']:.6f}", 
                    f"{time_for_train + time_for_validate:.4f}"
                ])
        
        if current_map > best_map:
            best_map = current_map
            epochs_without_improvement = 0
            torch.save(model.state_dict(), "best_faster_rcnn.pth")
            print(f"*** New Optimal Model Found! mAP: {best_map:.4f} - Saved Weights ***")
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement} epoch(s).")
        
        if epochs_without_improvement >= patience:
            print(f"Stopping early at epoch {epoch+1}. The optimal point was epoch {epoch+1 - patience}.")
            break

    print("Training Complete!")
