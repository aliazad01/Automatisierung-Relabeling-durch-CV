import os
import cv2
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import torch
torch.set_flush_denormal(True)

image_dir = r"C:\Users\AZADAL01\Documents\PA2\Praxis\Projekt\ocr\mehr_gedreht\input"

# Beispielbild, das nur um wenige Grad gedreht ist:
image_filename = "122220005130458104964293-202565163356_cls0_0_reingezoomt_cls0_0_reingezoomt.jpg"
image_path = os.path.join(image_dir, image_filename)
img = cv2.imread(image_path)

doc = DocumentFile.from_images(image_path, )

predictor = ocr_predictor(
    pretrained=True,
    det_arch="fast_base",
    reco_arch="parseq",
    assume_straight_pages=False,
    detect_orientation=True,     #  <-- This is the key parameter for orientation detection
    disable_crop_orientation=True,
    disable_page_orientation=True,
    straighten_pages=True        # <-- This will straighten the pages based on detected orientation
)  # .cuda().half() uncomment this line if we run on GPU

result = predictor(doc)

# Visualize the result
result

result.show()




