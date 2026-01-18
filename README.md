# Datasets

- Caltech Pedestrian Detection Benchmark: https://data.caltech.edu/records/f6rph-90m20
    - video data (.seq)
    - complex and with few pedestrians on average
- CityPersons: A Diverse Dataset for Pedestrian Detection: https://www.kaggle.com/datasets/hakurei/citypersons
    - subset of Cityscapes which only consists of person annotations
    - https://github.com/cvgroup-njust/CityPersons - additional useful repo
- Penn-Fudan: Database for Pedestrian Detection and Segmentation: https://www.cis.upenn.edu/~jshi/ped_html/
    - smaller dataset (170 images, 345 labeled pedestrians)

![alt text](image.png)

# Models

- YOLO - one-stage
- Faster R-CNN (with ResNet-50 / ResNet-101 backbone) - two-stage
- DETR (or Deformable DETR) — Transformer-based baseline?
- Custom simple model
- RetinaNet or RetinaNet + Soft-NMS?

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

- download caltech and citypersons from links above
- extract them into data/caltech and data/citypersons

### Setup

```
cd src
python prep.py
```
- this restructures the datasets (caltech and citypersons) into yolo-compatible standard coco format

### Training

```
python train.py <dataset> <model>
# look at script for options or run without params to see available
# for now only yolo model training works
```