import re
import shutil
import cv2
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import yaml
from data import read_seq, read_vbb

def prep_caltech(
    caltech_root='../data/caltech',
    output_root="..data/yolo/caltech",
    frame_stride=75,
):
    print('\nStarting caltech dataset prep...')
    caltech_root = Path(caltech_root)
    output_root = Path(output_root)

    for split in ["Train", "Test"]:
        img_out = output_root / "images" / ("train" if split == "Train" else "val")
        lbl_out = output_root / "labels" / ("train" if split == "Train" else "val")
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)
        set_dirs = sorted((caltech_root / split).iterdir())

        for set_dir in tqdm(set_dirs, desc=f"{split} sets", leave=False):            # Caltech: Train/setxx/setxx/*.seq
            inner_dir = set_dir / set_dir.name
            if not inner_dir.exists():
                continue

            for seq_file in sorted(inner_dir.glob("*.seq")):
                vbb_path = (
                    caltech_root
                    / "annotations"
                    / "annotations"
                    / set_dir.name
                    / seq_file.with_suffix(".vbb").name
                )

                if not vbb_path.exists():
                    continue

                frames = read_seq(seq_file, frame_stride)
                ann = read_vbb(vbb_path)

                for i in range(0, len(frames)):
                    boxes = ann.get(i, [])
                    # if len(boxes) == 0:
                    #     continue

                    frame_id, img = frames[i]
                    h, w = img.shape[:2]

                    stem = f"{set_dir.name}_{seq_file.stem}_{frame_id:06d}"
                    img_path = img_out / f"{stem}.jpg"
                    lbl_path = lbl_out / f"{stem}.txt"

                    cv2.imwrite(str(img_path), img[:, :, ::-1])

                    with open(lbl_path, "w") as f:
                        for x1, y1, x2, y2 in boxes:
                            cx = ((x1 + x2) / 2) / w
                            cy = ((y1 + y2) / 2) / h
                            bw = (x2 - x1) / w
                            bh = (y2 - y1) / h
                            f.write(f"0 {cx} {cy} {bw} {bh}\n")
        
        print('\tDone!')

def prep_citypersons(
    src_root="../data/citypersons/Citypersons",
    out_root="../data/yolo/citypersons",
):
    print('\nStarting citypersons dataset prep...')

    src_root = Path(src_root)
    out_root = Path(out_root)

    for split in ["train", "val", "test"]:
        (out_root / "images" / split).mkdir(parents=True, exist_ok=True)
        if split != "test":
            (out_root / "labels" / split).mkdir(parents=True, exist_ok=True)

        img_root = src_root / "images" / split

        for city_dir in tqdm(list(img_root.iterdir()), desc=f"Images {split}"):
            city = city_dir.name

            for img_path in city_dir.glob("*.png"):
                out_name = f"{city}_{img_path.name}"
                shutil.copy(
                    img_path,
                    out_root / "images" / split / out_name,
                )

                if split == "test":
                    continue

                lbl_path = (
                    src_root
                    / "labels"
                    / split
                    / city
                    / img_path.with_suffix(".txt").name
                )

                if lbl_path.exists():
                    shutil.copy(
                        lbl_path,
                        out_root / "labels" / split / out_name.replace(".png", ".txt"),
                    )
                else:
                    # YOLO allows empty label files
                    open(
                        out_root / "labels" / split / out_name.replace(".png", ".txt"),
                        "w",
                    ).close()

        print('\tDone!')

def prep_penn_fudan(
    src_root="../data/penn-fudan/PennFudanPed",
    out_root="../data/yolo/penn_fudan",
    split_ratio=(0.8, 0.1, 0.1),
):
    print("\nStarting penn-fudan dataset prep...")

    src_root = Path(src_root)
    out_root = Path(out_root)

    img_dir = src_root / "PNGImages"
    ann_dir = src_root / "Annotation"

    for split in ["train", "val", "test"]:
        (out_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    img_paths = sorted(img_dir.glob("*.png"))

    train_val, test = train_test_split(img_paths, test_size=split_ratio[2], random_state=11)
    train, val = train_test_split(
        train_val,
        test_size=split_ratio[1] / (split_ratio[0] + split_ratio[1]),
        random_state=11,
    )

    splits = {"train": train, "val": val, "test": test}

    bbox_pattern = re.compile(
        r"\((\d+),\s*(\d+)\)\s*-\s*\((\d+),\s*(\d+)\)"
    )

    for split_name, split_imgs in splits.items():
        for img_path in tqdm(split_imgs, desc=split_name):
            stem = img_path.stem
            ann_path = ann_dir / f"{stem}.txt"

            img = cv2.imread(str(img_path))
            h, w = img.shape[:2] # type: ignore

            out_img = out_root / "images" / split_name / img_path.name
            cv2.imwrite(str(out_img), img) # type: ignore

            out_lbl = out_root / "labels" / split_name / f"{stem}.txt"

            if not ann_path.exists():
                open(out_lbl, "w").close()
                continue

            labels = []

            with open(ann_path) as f:
                for line in f:
                    if "Bounding box" in line:
                        match = bbox_pattern.search(line)
                        if not match:
                            continue

                        x1, y1, x2, y2 = map(float, match.groups())

                        cx = ((x1 + x2) / 2) / w
                        cy = ((y1 + y2) / 2) / h
                        bw = (x2 - x1) / w
                        bh = (y2 - y1) / h

                        labels.append(f"0 {cx} {cy} {bw} {bh}")

            with open(out_lbl, "w") as f:
                f.write("\n".join(labels))

    dataset_yaml = {
        "path": str(out_root),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "person"},
    }

    with open(out_root / "dataset.yaml", "w") as f:
        yaml.dump(dataset_yaml, f)

    print("\tDone!")


if __name__ == '__main__':
    # prep_caltech()
    # prep_citypersons()
    prep_penn_fudan()
