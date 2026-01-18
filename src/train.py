import sys

from tqdm import tqdm

from data import *
from models import *
from utils import *
from torch.utils.data import DataLoader

def run_pipeline(dataset_type: DatasetType, model_type: ModelType):
    print('Starting training pipeline...')
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"

    try:
        # load and prepare dataset
        print(f'Preparing dataset "{dataset_type.value}"...')
        if model_type != ModelType.YOLO:
            dataloader = load_dataset(dataset_type)

        # load and prepare model
        print(f'\tPreparing model "{model_type.value}"...')
        if model_type == ModelType.YOLO:
            model = load_yolo_model()
            model.train(
                data=f'../data/yolo/{dataset_type.value}/dataset.yaml',
                epochs=1,
                batch=8,
                device='0'
            )


        elif model_type == ModelType.DETR:
            exit(1)


        elif model_type == ModelType.RCNN:
            model = load_rcnn_model(device)

            model.train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

            for epoch in range(1):
                for images, targets in dataloader:
                    images = [img.to(device) for img in images]
                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                    loss_dict = model(images, targets)
                    losses = sum(loss for loss in loss_dict.values())

                    optimizer.zero_grad()
                    losses.backward() # type: ignore
                    optimizer.step()

                    print("Loss:", losses.item()) # type: ignore


        elif model_type == ModelType.CUSTOM:
            model = load_custom_model(device=device, num_classes=2)

            model.train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

            num_epochs = 1
            for epoch in range(num_epochs):
                epoch_loss = 0.0
                num_batches = 0

                for images, targets in dataloader:
                    images = [img.to(device) for img in images]
                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                    # Forward pass - model returns loss dict in training mode
                    loss_dict = model(images, targets)
                    loss = loss_dict['loss']

                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()
                    num_batches += 1

                    print(f"Epoch [{epoch+1}/{num_epochs}] Batch Loss: {loss.item():.4f} "
                          f"(rpn_cls: {loss_dict['loss_rpn_cls'].item():.4f}, "
                          f"rpn_reg: {loss_dict['loss_rpn_reg'].item():.4f}, "
                          f"roi_cls: {loss_dict['loss_roi_cls'].item():.4f}, "
                          f"roi_reg: {loss_dict['loss_roi_reg'].item():.4f})")

                avg_loss = epoch_loss / max(num_batches, 1)
                print(f"Epoch [{epoch+1}/{num_epochs}] Average Loss: {avg_loss:.4f}")

            # Ensure models directory exists before saving
            import os
            models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
            os.makedirs(models_dir, exist_ok=True)
            model_path = os.path.join(models_dir, f'custom_{dataset_type.value}.pth')
            torch.save(model.state_dict(), model_path)
            print(f"Model saved to {model_path}")

    except Exception as e:
        print(e)

if __name__ == '__main__':
    import torch
    torch.cuda.empty_cache()
    import gc
    gc.collect()

    # parse system parameters
    args = sys.argv
    if len(args) < 3:
        print('Usage: python train.py <dataset> <model>')
        print(f'<dataset> options:\n\t\t' + '\n\t\t'.join([d.value for d in DatasetType]))
        print(f'<model> options:\n\t\t' + '\n\t\t'.join([m.value for m in ModelType]))
        exit(1)

    # run
    run_pipeline(DatasetType(args[1]), ModelType(args[2]))