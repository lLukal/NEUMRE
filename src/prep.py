import shutil
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
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
        
        print('Done!')

def prep_citypersons(
    src_root="../data/citypersons/Citypersons",
    out_root="../data/yolo/citypersons",
):
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

if __name__ == '__main__':
    # prep_caltech()
    prep_citypersons()
