from pathlib import Path
import os
import shutil
import cv2
from tqdm import tqdm
import numpy as np
import yaml
from ultralytics import YOLO

from pathlib import Path
from itertools import islice
import os
import shutil
from pathlib import Path
from typing import Iterable, List, Optional, Tuple
import cv2
import numpy as np
from ultralytics import YOLO
import torch
from tqdm import tqdm
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Die Objektklassen die wir erkennen wollen.
CLASSES = [
    "dhl_shippingLabel"
]
# Hier werden die Ergebnisse gespeichert.
INTERFERENCE_DIR = r'C:\Users\AZADAL01\Documents\PA2\Praxis\Projekt\yolo\model_testen\output_test2'
# Das verwendete Modell.
CHECKPOINT_PATH=r"C:\Users\AZADAL01\Documents\PA2\Praxis\Projekt\yolo\model_testen\best_2025-09-15_15-01-35.pt"
# Hier liegen die Bilder die wir auswerten wollen.
ROOT_TEST = r"C:\Users\AZADAL01\Documents\PA2\Praxis\Projekt\yolo\model_testen\data"

# Clean output dir
if os.path.exists(INTERFERENCE_DIR):
    shutil.rmtree(INTERFERENCE_DIR)
os.makedirs(INTERFERENCE_DIR, exist_ok=True)



train_imgs_dir = os.path.join(ROOT_TEST, "images", "train")
val_imgs_dir = os.path.join(ROOT_TEST, "images", "val")
test_imgs_dir = os.path.join(ROOT_TEST, "images", "test")

# WORK_DIR = "./runs"           # Where YOLO will save runs
DATA_YAML = os.path.join(ROOT_TEST, "data.yaml")
MODEL_YAML = "yolo11l.yaml"  
EXPER_NAME = "dhl_label_detection"


# ------------------------
# WRITE data.yaml
# ------------------------
data_dict = {
    "train": train_imgs_dir,
    "val": val_imgs_dir,
    "test": test_imgs_dir,
    "nc": len(CLASSES),
    "names": CLASSES
}

os.makedirs(ROOT_TEST, exist_ok=True)
with open(DATA_YAML, "w") as f:
    yaml.dump(data_dict, f, sort_keys=False)
print(f"[INFO] Wrote dataset YAML to {DATA_YAML}")


# ------------------------
# GPU / DEVICE CHECK
# ------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Training on device: {device}")




# Load model
model = YOLO(CHECKPOINT_PATH)

# ----------------------------
# 1) Run evaluation on test set
# ----------------------------
# YOLO expects dataset in COCO or YOLO format: images + labels/
# Here ROOT_TEST should contain images, and labels must be in ROOT_TEST/labels
metrics = model.val(
    data=DATA_YAML,
    split="test",                 # ensure it uses your test set
    save_json=True,               # save COCO-style metrics
    project=INTERFERENCE_DIR,     # save results under this dir
    name="metrics"                # subfolder name
)

# Save metrics summary to a txt file
metrics_path = Path(INTERFERENCE_DIR) / "metrics" / "metrics_summary.txt"
with open(metrics_path, "w") as f:
    f.write(str(metrics))  # metrics is a Results object with dict-like values

print(f"Validation metrics saved to: {metrics_path}")

# ----------------------------
# 2) Run inference & save annotated images + labels
# ----------------------------
image_paths = sorted([
    p for p in Path(test_imgs_dir).iterdir()
    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
])

out_images = Path(INTERFERENCE_DIR) / "images" / "annotated"
out_labels = Path(INTERFERENCE_DIR) / "labels"
out_images.mkdir(parents=True, exist_ok=True)
out_labels.mkdir(parents=True, exist_ok=True)

pbar = tqdm(total=len(image_paths), desc="Processing images", unit="img")
for p in image_paths:
    res = model.predict(
        source=str(p),
        conf=0.25, iou=0.45, max_det=50,
        device="cpu", save=False, verbose=False
    )[0]

    # Save YOLO-format labels
    labels_path = out_labels / f"{p.stem}.txt"
    if getattr(res, "boxes", None) is not None and len(res.boxes) > 0:
        clss = res.boxes.cls.detach().cpu().numpy().astype(int).tolist()
        xywhn = res.boxes.xywhn.detach().cpu().numpy().tolist()
        with labels_path.open("w", encoding="utf-8") as f:
            for c, (xc, yc, w, h) in zip(clss, xywhn):
                f.write(f"{c} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
    else:
        labels_path.touch()

    # Save annotated image
    ann = res.plot()
    cv2.imwrite(str(out_images / f"{p.stem}{p.suffix}"), ann)

    del ann
    pbar.update(1)

pbar.close()



# ----------------------------
# 3) Calculate Average IoU and Loss
# ----------------------------
def xywhn_to_xyxy(box, img_w, img_h):
    """Convert normalized xywh box to absolute xyxy (x1, y1, x2, y2)."""
    x_c, y_c, w, h = box
    x1 = (x_c - w / 2) * img_w
    y1 = (y_c - h / 2) * img_h
    x2 = (x_c + w / 2) * img_w
    y2 = (y_c + h / 2) * img_h
    return [x1, y1, x2, y2]


def compute_iou(boxA, boxB):
    """Compute IoU between two boxes in xyxy format."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)


ious = []

# Iterate over test images again
for img_path in tqdm(image_paths, desc="Calculating IoU", unit="img"):
    label_path = Path(ROOT_TEST) / "labels" / "test" / f"{img_path.stem}.txt"
    pred_path = out_labels / f"{img_path.stem}.txt"

    if not label_path.exists() or not pred_path.exists():
        continue

    # Load ground truth boxes
    with open(label_path, "r") as f:
        gt_boxes = [list(map(float, line.strip().split()[1:])) for line in f.readlines()]

    # Load predicted boxes
    with open(pred_path, "r") as f:
        pred_boxes = [list(map(float, line.strip().split()[1:])) for line in f.readlines()]

    # Skip if no detections or labels
    if not gt_boxes or not pred_boxes:
        continue

    img = cv2.imread(str(img_path))
    img_h, img_w = img.shape[:2]

    # Convert normalized to pixel coordinates
    gt_boxes_xyxy = [xywhn_to_xyxy(b, img_w, img_h) for b in gt_boxes]
    pred_boxes_xyxy = [xywhn_to_xyxy(b, img_w, img_h) for b in pred_boxes]

    # Compute IoU for each predicted vs ground truth (max match)
    for pb in pred_boxes_xyxy:
        best_iou = max([compute_iou(pb, gb) for gb in gt_boxes_xyxy])
        ious.append(best_iou)

# Compute mean IoU and loss
if ious:
    avg_iou = np.mean(ious)
    avg_loss = 1 - avg_iou
else:
    avg_iou = 0.0
    avg_loss = 1.0

print(f"\n[RESULTS] Average IoU: {avg_iou:.4f}")
print(f"[RESULTS] Average Loss (1 - IoU): {avg_loss:.4f}")

# Save results
iou_results_path = Path(INTERFERENCE_DIR) / "metrics" / "iou_results.txt"
with open(iou_results_path, "w") as f:
    f.write(f"Average IoU: {avg_iou:.4f}\n")
    f.write(f"Average Loss (1 - IoU): {avg_loss:.4f}\n")

print(f"IoU results saved to: {iou_results_path}")
