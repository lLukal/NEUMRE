# Datasets

- Caltech Pedestrian Detection Benchmark: https://data.caltech.edu/records/f6rph-90m20
    - video data (.seq)
    - complex and with few pedestrians on average
- CityPersons: A Diverse Dataset for Pedestrian Detection: https://www.kaggle.com/datasets/hakurei/citypersons
    - subset of Cityscapes which only consists of person annotations
    - https://github.com/cvgroup-njust/CityPersons - additional useful repo
- Penn-Fudan: Database for Pedestrian Detection and Segmentation: https://www.cis.upenn.edu/~jshi/ped_html/
    - smaller dataset (170 images, 345 labeled pedestrians)

# Models

- YOLO
- Faster R-CNN (with ResNet-50 / ResNet-101 backbone)
- DETR (or Deformable DETR)
- Custom model architecture

# Instructions

### Python Environment

- recommended: use venv
```
python -m venv venv
source ./venv/bin/activate
```
- alternate: use docker with tensorflow cuda image (not tested)
```
sudo docker compose up # or just docker compose up
# or use something like dev containers extension in vscode to access the container
```

### Datasets

- download penn_fudan and citypersons from links above
- extract them into data/penn-fudan and data/citypersons

### Setup

```
cd src
python prep.py
```
- this restructures the datasets into yolo-compatible standard coco format for further use

### Training

```
python train.py
# run without params to see available options
# special script for rcnn training in different structure 
#   python train_rcnn.py
```

### Data Exploration (visualization of detections after training)

```
python explore.py
# run without params to see available options
```

### Evaluation

```
python eval.py
# run without params to see available options
```

### Other Modules (not used directly)

```
# overfit_test.py
#   test specific to custom model
# models.py
#   contains definitions of model
# data.py
#   contains data handling functions and tools
# utils.py
#   other utilities used by other modules
# trained_models
#   contains saved .pt or .pth models after training, used for explore.py and visualize.py
```
