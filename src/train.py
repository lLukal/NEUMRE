import csv
from datetime import timedelta
import os
import sys
import time

from tqdm import tqdm

from data import *
from eval import evaluate_detr, evaluate_rcnn
from models import *
from utils import *

#region HELPERS

def train_rcnn_helper(model, train_loader, device, optimizer):
    model.train()
    train_loss = 0
    
    for images, targets in train_loader:
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward() # type: ignore
        optimizer.step()

        train_loss += losses.item() # type: ignore
    
    return train_loss / len(train_loader)

#endregion

#region API
#endregion

def train_yolo(model, dataset_type):
    model.train(
        data=f'../data/yolo/{dataset_type.value}/dataset.yaml',
        epochs=50,
        batch=4,
        imgsz=1024
    )

def train_detr(model, train_dataloader, val_dataloader, save_dir, device):
    base_lr = 1e-4
    backbone_lr = 1e-5
    num_epochs = 50
    best_map = 0.0

    # checkpoint
    resume_weights = f'{save_dir}/detr_last.pth'

    if os.path.exists(resume_weights):
        print(f"\tFound checkpoint: {resume_weights}. Loading weights for resume...")
        model.load_state_dict(torch.load(resume_weights, map_location=device)) # type: ignore
        base_lr = 5e-5
        backbone_lr = 5e-6
    else:
        print("\tNo checkpoint found - starting training from scratch...")

    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n], "lr": base_lr}, # type: ignore
        {"params": [p for n, p in model.named_parameters() if "backbone" in n], "lr": backbone_lr}, # type: ignore
    ]
    optimizer = torch.optim.AdamW(param_dicts, weight_decay=1e-4)

    print(f"Starting training for {num_epochs} epochs with base LR: {base_lr}...")

    for epoch in range(num_epochs):
        model.train() # type: ignore
        epoch_loss = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")

        for images, targets in progress_bar:
            images = images.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(pixel_values=images, labels=targets) # type: ignore
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1) # type: ignore
            optimizer.step()

            epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        metrics = evaluate_detr(model, val_dataloader, device, output_dir=save_dir, epoch_num=epoch+1)
        
        current_map = metrics['map'].item()
        current_map_50 = metrics['map_50'].item()
        
        print(f"\tEpoch {epoch+1} Results: Loss={epoch_loss/len(train_dataloader):.4f} | mAP={current_map:.4f} | mAP50={current_map_50:.4f}")

        # save last model
        torch.save(model.state_dict(), f'{save_dir}/detr_last.pth') # type: ignore

        # save best model
        if current_map > best_map:
            print(f"\tNew best mAP! ({best_map:.4f} -> {current_map:.4f}). Saving model...")
            best_map = current_map
            torch.save(model.state_dict(), f'{save_dir}/detr_best.pth') # type: ignore

def train_rcnn(model, train_dataloader, val_dataloader, save_dir, device):
    epochs_without_improvement = 0
    best_map = -1

    num_epochs = 50
    patience = 5
    log_file = "./runs/rcnn/training_log.csv"

    with open(log_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Train_Loss", "Val_Loss", "mAP", "mAP_50", "Learning_Rate", "Time_For_Epoch"])

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=4, gamma=0.1
    )

    print(f"Starting training for {num_epochs} epochs...")

    for epoch in range(num_epochs):
        start_time = time.time()
        avg_train_loss = train_rcnn_helper(model, train_dataloader, device, optimizer)
        validate_start = time.time()
        avg_val_loss, map_scores = evaluate_rcnn(model, val_dataloader, device)
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
            f"Epoch [{epoch+1}/{num_epochs}] | "
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
            print(f"\tNew Optimal Model Found! mAP: {best_map:.4f} - Saved Weights ***")
        else:
            epochs_without_improvement += 1
            print(f"\tNo improvement for {epochs_without_improvement} epoch(s).")
        
        if epochs_without_improvement >= patience:
            print(f"\tStopping early at epoch {epoch+1}. The optimal point was epoch {epoch+1 - patience}.")
            break

    print("Training Complete!")

def train_custom(model, train_dataloader, val_dataloader, save_dir, device, overfit_test=False):
    """Train custom Faster R-CNN-style model with mAP evaluation."""
    from eval import evaluate_custom
    
    # Overfit test: use small subset and more epochs
    if overfit_test:
        num_epochs = 50
        learning_rate = 1e-3  # Higher LR for faster overfitting
        print("\n*** OVERFIT TEST MODE: Training on small subset ***\n")
    else:
        num_epochs = 10
        learning_rate = 3e-4
    
    weight_decay = 1e-4
    patience = 100 if overfit_test else 10  # Don't early stop during overfit test
    
    best_map = -1
    epochs_without_improvement = 0
    log_file = f"{save_dir}/custom_training_log.csv"
    
    os.makedirs(save_dir, exist_ok=True)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    # Initialize CSV log
    with open(log_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Train_Loss", "Val_Loss", "mAP", "mAP_50", "mAP_75", "Learning_Rate", "Time"])
    
    print(f"Starting training for {num_epochs} epochs...")
    print(f"Learning rate: {learning_rate}, Weight decay: {weight_decay}")
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Training
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for images, targets in progress_bar:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            loss = loss_dict['loss']

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
            
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'rpn_cls': f"{loss_dict['loss_rpn_cls'].item():.4f}",
                'roi_cls': f"{loss_dict['loss_roi_cls'].item():.4f}"
            })

        avg_train_loss = epoch_loss / max(num_batches, 1)
        
        # Validation with mAP
        val_loss, map_results = evaluate_custom(model, val_dataloader, device)
        
        current_map = map_results['map'].item()
        current_map_50 = map_results['map_50'].item()
        current_map_75 = map_results['map_75'].item()
        
        # Update learning rate scheduler
        lr_scheduler.step(current_map)
        current_lr = optimizer.param_groups[0]['lr']
        
        epoch_time = time.time() - start_time
        td = timedelta(seconds=epoch_time)
        
        print(f"Epoch [{epoch+1}/{num_epochs}] | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"mAP: {current_map:.4f} | "
              f"mAP@50: {current_map_50:.4f} | "
              f"mAP@75: {current_map_75:.4f} | "
              f"LR: {current_lr:.6f} | "
              f"Time: {td}")
        
        # Log to CSV
        with open(log_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1,
                f"{avg_train_loss:.4f}",
                f"{val_loss:.4f}",
                f"{current_map:.4f}",
                f"{current_map_50:.4f}",
                f"{current_map_75:.4f}",
                f"{current_lr:.6f}",
                f"{epoch_time:.2f}"
            ])
        
        # Save last model
        torch.save(model.state_dict(), f"{save_dir}/custom_last.pth")
        
        # Save best model
        if current_map > best_map:
            best_map = current_map
            epochs_without_improvement = 0
            torch.save(model.state_dict(), f"{save_dir}/custom_best.pth")
            print(f"\tNew best mAP! ({best_map:.4f}) - Model saved.")
        else:
            epochs_without_improvement += 1
            print(f"\tNo improvement for {epochs_without_improvement} epoch(s).")
        
        # Early stopping
        if epochs_without_improvement >= patience:
            print(f"\tEarly stopping at epoch {epoch+1}. Best mAP: {best_map:.4f}")
            break
    
    print(f"Training complete! Best mAP: {best_map:.4f}")
    return best_map

#region MAIN

def run_pipeline(dataset_type: DatasetType, model_type: ModelType, overfit_test=False):
    print('Starting training pipeline...')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    save_dir = '../trained_models'

    try:
        # load and prepare dataset
        print(f'Preparing dataset "{dataset_type.value}"...')
        if model_type != ModelType.YOLO:
            train_dataloader = load_dataset(dataset_type, model_type, split='train') # type: ignore
            val_dataloader = load_dataset(dataset_type, model_type, split='val') # type: ignore
            
            # For overfit test, use only first 10 batches
            if overfit_test:
                print("*** OVERFIT TEST: Using only 10 batches ***")
                train_subset = list(train_dataloader)[:10]
                val_subset = list(val_dataloader)[:5]
                train_dataloader = train_subset
                val_dataloader = val_subset

        # load and train model
        print(f'Preparing model "{model_type.value}"...')
        if model_type == ModelType.YOLO:
            model = load_yolo_model('yolov8n')
            train_yolo(model, dataset_type)


        elif model_type == ModelType.DETR:
            model = load_detr_model(device)
            train_detr(model, train_dataloader, val_dataloader, save_dir, device)


        elif model_type == ModelType.RCNN:
            model = load_rcnn_model(device)
            train_rcnn(model, train_dataloader, val_dataloader, save_dir, device)


        elif model_type == ModelType.CUSTOM:
            model = load_custom_model(device=device, num_classes=2)
            train_custom(model, train_dataloader, val_dataloader, save_dir, device, overfit_test=overfit_test)

    except Exception as e:
        import traceback
        print(f"ERROR: {e}")
        traceback.print_exc()

if __name__ == '__main__':
    import torch
    torch.cuda.empty_cache()
    import gc
    gc.collect()

    # parse system parameters
    args = sys.argv
    if len(args) < 3:
        print('Usage: python train.py <dataset> <model> [--overfit]')
        print(f'<dataset> - options:\n\t\t' + '\n\t\t'.join([d.value for d in DatasetType]))
        print(f'<model> - options:\n\t\t' + '\n\t\t'.join([m.value for m in ModelType]))
        print('--overfit: Run overfit test on small subset')
        exit(1)

    # Check for overfit flag
    overfit_test = '--overfit' in args

    # run
    run_pipeline(DatasetType(args[1]), ModelType(args[2]), overfit_test=overfit_test)